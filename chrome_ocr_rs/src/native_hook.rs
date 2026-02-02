//! Hook TFLite recognition model input capture
//!
//! chrome_screen_ai.dll statically links TFLite with LTO.
//! Both Interpreter::Invoke() and Subgraph::Invoke() are inlined at
//! all internal call sites, so hooking them is impossible.
//!
//! Instead, we hook the real AllocateTensors (RVA 0x013cb730) which
//! has 10 internal callers. The C API wrapper does:
//!   TfLiteInterpreterAllocateTensors (0x00f37ab0):
//!     mov rcx, [rcx+0x18]    // rcx = interpreter->impl
//!     jmp 0x013cb730          // real_allocate(impl)
//!
//! The actual input DATA is written AFTER AllocateTensors but before
//! the (inlined) Invoke. We use a two-phase capture:
//!   Phase 1 (on AllocateTensors): detect recognition model shape,
//!           remember the tensor data pointer.
//!   Phase 2 (on the next AllocateTensors call, or on PerformOCR return):
//!           the data pointer still holds valid input from the completed
//!           inference, so we save it then.

use anyhow::{anyhow, Result};
use libloading::Library;
use once_cell::sync::OnceCell;
use retour::GenericDetour;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

// =============================================================================
// TFLite C API types and signatures
// =============================================================================

#[repr(C)]
pub struct TfLiteInterpreter {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct TfLiteTensor {
    _opaque: [u8; 0],
}

type TfLiteStatus = i32;

/// Real AllocateTensors takes impl_ptr (what lives at TfLiteInterpreter+0x18)
type RealAllocateTensorsFn = unsafe extern "C" fn(impl_ptr: *mut c_void) -> TfLiteStatus;

type GetInputTensorFn = unsafe extern "C" fn(
    interpreter: *const TfLiteInterpreter,
    input_index: i32,
) -> *const TfLiteTensor;
type TensorNumDimsFn = unsafe extern "C" fn(tensor: *const TfLiteTensor) -> i32;
type TensorDimFn = unsafe extern "C" fn(tensor: *const TfLiteTensor, dim_index: i32) -> i32;
type TensorDataFn = unsafe extern "C" fn(tensor: *const TfLiteTensor) -> *const u8;
type TensorByteSizeFn = unsafe extern "C" fn(tensor: *const TfLiteTensor) -> usize;

// RVAs determined from disassembly of chrome_screen_ai.dll v140.14
const C_API_ALLOCATE_RVA: usize = 0x00f37ab0;
const REAL_ALLOCATE_RVA: usize = 0x013cb730;

// Recognition model input dimensions
const REC_HEIGHT: i32 = 32;
const REC_WIDTH: i32 = 168;
const REC_CHANNELS: i32 = 1;

// =============================================================================
// Fake TfLiteInterpreter layout
// =============================================================================
//
// The C API functions (GetInputTensor, etc.) read impl_ptr at offset 0x18:
//   mov rax, [rcx+0x18]
// We build a stack struct with impl_ptr placed there so the C API works.

#[repr(C, align(8))]
struct FakeInterpreter {
    _pad: [u8; 0x18],
    impl_ptr: *mut c_void,
    _tail: [u8; 8],
}

impl FakeInterpreter {
    fn new(impl_ptr: *mut c_void) -> Self {
        FakeInterpreter {
            _pad: [0u8; 0x18],
            impl_ptr,
            _tail: [0u8; 8],
        }
    }
    fn as_ptr(&self) -> *const TfLiteInterpreter {
        self as *const _ as *const TfLiteInterpreter
    }
}

// =============================================================================
// Global state
// =============================================================================

static OUTPUT_DIR: OnceCell<PathBuf> = OnceCell::new();
static CALL_COUNTER: AtomicUsize = AtomicUsize::new(0);
static CAPTURE_COUNTER: AtomicUsize = AtomicUsize::new(0);

static GET_INPUT_TENSOR: OnceCell<GetInputTensorFn> = OnceCell::new();
static TENSOR_NUM_DIMS: OnceCell<TensorNumDimsFn> = OnceCell::new();
static TENSOR_DIM: OnceCell<TensorDimFn> = OnceCell::new();
static TENSOR_DATA: OnceCell<TensorDataFn> = OnceCell::new();
static TENSOR_BYTE_SIZE: OnceCell<TensorByteSizeFn> = OnceCell::new();

/// Pending capture: stores info from AllocateTensors to be flushed later
/// when the inference has completed and the tensor buffer holds valid data.
struct PendingCapture {
    batch: i32,
    data_ptr: *const u8,
    byte_size: usize,
}

unsafe impl Send for PendingCapture {}

static PENDING: Mutex<Vec<PendingCapture>> = Mutex::new(Vec::new());

// =============================================================================
// Hook: real AllocateTensors
// =============================================================================

static DETOUR_ALLOC: OnceCell<GenericDetour<RealAllocateTensorsFn>> = OnceCell::new();

unsafe extern "C" fn hooked_allocate_tensors(impl_ptr: *mut c_void) -> TfLiteStatus {
    let _call_idx = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);

    // Flush any pending captures from previously completed inference
    flush_pending_captures();

    // Call original AllocateTensors
    let detour = DETOUR_ALLOC.get().expect("detour must be set");
    let status = detour.call(impl_ptr);

    // After AllocateTensors succeeds, check if this is the recognition model
    if status == 0 {
        check_and_register(impl_ptr);
    }

    status
}

unsafe fn check_and_register(impl_ptr: *mut c_void) {
    let get_input = match GET_INPUT_TENSOR.get() {
        Some(f) => f,
        None => return,
    };
    let num_dims = match TENSOR_NUM_DIMS.get() {
        Some(f) => f,
        None => return,
    };
    let dim_fn = match TENSOR_DIM.get() {
        Some(f) => f,
        None => return,
    };
    let data_fn = match TENSOR_DATA.get() {
        Some(f) => f,
        None => return,
    };
    let byte_size_fn = match TENSOR_BYTE_SIZE.get() {
        Some(f) => f,
        None => return,
    };

    let fake = FakeInterpreter::new(impl_ptr);
    let interp = fake.as_ptr();

    let tensor = get_input(interp, 0);
    if tensor.is_null() {
        return;
    }

    let ndims = num_dims(tensor);
    if ndims != 4 {
        return;
    }

    let d1 = dim_fn(tensor, 1);
    let d2 = dim_fn(tensor, 2);
    let d3 = dim_fn(tensor, 3);

    if d1 != REC_HEIGHT || d2 != REC_WIDTH || d3 != REC_CHANNELS {
        return;
    }

    let batch = dim_fn(tensor, 0);
    let data_ptr = data_fn(tensor);
    let byte_size = byte_size_fn(tensor);

    if let Ok(mut pending) = PENDING.lock() {
        pending.push(PendingCapture {
            batch,
            data_ptr,
            byte_size,
        });
    }
}

/// Flush pending captures — read input tensor data that was populated
/// during the (now-completed) inference.
fn flush_pending_captures() {
    let captures: Vec<PendingCapture> = {
        match PENDING.lock() {
            Ok(mut pending) => std::mem::take(&mut *pending),
            Err(_) => return,
        }
    };

    if captures.is_empty() {
        return;
    }

    let output_dir = match OUTPUT_DIR.get() {
        Some(d) => d,
        None => return,
    };

    for cap in captures {
        let pixels_per_item = (REC_HEIGHT as usize) * (REC_WIDTH as usize);
        let expected = (cap.batch as usize) * pixels_per_item;

        if cap.data_ptr.is_null() || cap.byte_size < expected {
            continue;
        }

        let counter = CAPTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
        println!("  [hook] Capture #{}: batch={}", counter, cap.batch);

        for b in 0..cap.batch as usize {
            let offset = b * pixels_per_item;
            let pixel_data =
                unsafe { std::slice::from_raw_parts(cap.data_ptr.add(offset), pixels_per_item) };

            let img = image::GrayImage::from_raw(
                REC_WIDTH as u32,
                REC_HEIGHT as u32,
                pixel_data.to_vec(),
            );

            if let Some(img) = img {
                let filename = format!("rec_input_{:05}_{:02}.png", counter, b);
                let path = output_dir.join(&filename);
                if let Err(e) = img.save(&path) {
                    eprintln!("  [hook] Failed to save {}: {}", path.display(), e);
                }
            }
        }
    }
}

/// Called externally after PerformOCR returns to flush the last batch.
pub fn flush_final_captures() {
    flush_pending_captures();
}

// =============================================================================
// Public API
// =============================================================================

pub struct NativeHook {
    _dll_lib: Library,
}

impl NativeHook {
    /// Install the hook on the real internal AllocateTensors function.
    ///
    /// `chrome_dll_path` must point to the already-loaded chrome_screen_ai.dll.
    /// Must be called **after** the DLL has been loaded and initialised.
    pub fn install(chrome_dll_path: &Path, output_dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&output_dir)?;

        OUTPUT_DIR
            .set(output_dir.clone())
            .map_err(|_| anyhow!("OUTPUT_DIR already set"))?;

        let dll_lib = unsafe { Library::new(chrome_dll_path) }
            .map_err(|e| anyhow!("Failed to open chrome_screen_ai.dll: {}", e))?;

        unsafe {
            // Resolve C API helper function pointers
            let get_input_ptr: libloading::Symbol<GetInputTensorFn> =
                dll_lib.get(b"TfLiteInterpreterGetInputTensor\0")?;
            let num_dims_ptr: libloading::Symbol<TensorNumDimsFn> =
                dll_lib.get(b"TfLiteTensorNumDims\0")?;
            let dim_ptr: libloading::Symbol<TensorDimFn> = dll_lib.get(b"TfLiteTensorDim\0")?;
            let data_ptr: libloading::Symbol<TensorDataFn> = dll_lib.get(b"TfLiteTensorData\0")?;
            let byte_size_ptr: libloading::Symbol<TensorByteSizeFn> =
                dll_lib.get(b"TfLiteTensorByteSize\0")?;

            GET_INPUT_TENSOR
                .set(*get_input_ptr)
                .map_err(|_| anyhow!("already set"))?;
            TENSOR_NUM_DIMS
                .set(*num_dims_ptr)
                .map_err(|_| anyhow!("already set"))?;
            TENSOR_DIM
                .set(*dim_ptr)
                .map_err(|_| anyhow!("already set"))?;
            TENSOR_DATA
                .set(*data_ptr)
                .map_err(|_| anyhow!("already set"))?;
            TENSOR_BYTE_SIZE
                .set(*byte_size_ptr)
                .map_err(|_| anyhow!("already set"))?;

            // Compute real AllocateTensors address from DLL base
            let c_api_alloc: libloading::Symbol<RealAllocateTensorsFn> =
                dll_lib.get(b"TfLiteInterpreterAllocateTensors\0")?;
            let c_api_addr = *c_api_alloc as *const () as usize;
            let dll_base = c_api_addr - C_API_ALLOCATE_RVA;
            let real_alloc_addr = dll_base + REAL_ALLOCATE_RVA;

            let real_alloc_fn: RealAllocateTensorsFn =
                std::mem::transmute(real_alloc_addr as *const ());

            let detour = GenericDetour::<RealAllocateTensorsFn>::new(
                real_alloc_fn,
                hooked_allocate_tensors,
            )?;
            detour.enable()?;

            DETOUR_ALLOC
                .set(detour)
                .map_err(|_| anyhow!("DETOUR already set"))?;
        }

        println!(
            "  [hook] Rec input capture active, output: {}",
            output_dir.display()
        );

        Ok(NativeHook { _dll_lib: dll_lib })
    }
}

impl Drop for NativeHook {
    fn drop(&mut self) {
        flush_pending_captures();
    }
}
