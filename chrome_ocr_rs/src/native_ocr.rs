//! Native Chrome Screen AI OCR implementation
//! Directly calls chrome_screen_ai.dll's PerformOCR with correct SkBitmap ABI

use anyhow::{anyhow, Result};
use image::GrayImage;
use libloading::{Library, Symbol};
use std::ffi::CString;
use std::path::{Path, PathBuf};

use crate::native_hook;

// =============================================================================
// SkBitmap ABI structures (reverse engineered from chrome_screen_ai.dll)
// =============================================================================

// SkColorType enum values
#[allow(dead_code)]
const SK_COLOR_TYPE_UNKNOWN: i32 = 0;
#[allow(dead_code)]
const SK_COLOR_TYPE_ALPHA_8: i32 = 1;
#[allow(dead_code)]
const SK_COLOR_TYPE_RGB_565: i32 = 2;
#[allow(dead_code)]
const SK_COLOR_TYPE_ARGB_4444: i32 = 3;
#[allow(dead_code)]
const SK_COLOR_TYPE_RGBA_8888: i32 = 4;
#[allow(dead_code)]
const SK_COLOR_TYPE_RGB_888X: i32 = 5;
const SK_COLOR_TYPE_BGRA_8888: i32 = 6;
#[allow(dead_code)]
const SK_COLOR_TYPE_GRAY_8: i32 = 14;

// SkAlphaType enum values
#[allow(dead_code)]
const SK_ALPHA_TYPE_UNKNOWN: i32 = 0;
const SK_ALPHA_TYPE_OPAQUE: i32 = 1;
#[allow(dead_code)]
const SK_ALPHA_TYPE_PREMUL: i32 = 2;
#[allow(dead_code)]
const SK_ALPHA_TYPE_UNPREMUL: i32 = 3;

/// SkImageInfo - image metadata (24 bytes)
/// Offsets within SkBitmap: colorSpace=0x18, colorType=0x20, alphaType=0x24, width=0x28, height=0x2C
#[repr(C)]
#[derive(Debug, Clone)]
struct SkImageInfo {
    color_space: *const std::ffi::c_void, // +0x00: sk_sp<SkColorSpace> (8 bytes)
    color_type: i32,                      // +0x08: SkColorType (4 bytes)
    alpha_type: i32,                      // +0x0C: SkAlphaType (4 bytes)
    width: i32,                           // +0x10: width (4 bytes)
    height: i32,                          // +0x14: height (4 bytes)
}

/// SkPixmap - lightweight pixel access (40 bytes)
/// Offsets within SkBitmap: pixels=0x08, rowBytes=0x10, info=0x18
#[repr(C)]
#[derive(Debug, Clone)]
struct SkPixmap {
    pixels: *const u8, // +0x00: pixel data pointer (8 bytes)
    row_bytes: usize,  // +0x08: bytes per row (8 bytes)
    info: SkImageInfo, // +0x10: image info (24 bytes)
}

/// SkBitmap - main bitmap class (56 bytes)
#[repr(C)]
#[derive(Debug, Clone)]
struct SkBitmap {
    pixel_ref: *const std::ffi::c_void, // +0x00: sk_sp<SkPixelRef> (8 bytes)
    pixmap: SkPixmap,                   // +0x08: embedded SkPixmap (40 bytes)
    flags: u8,                          // +0x30: flags (1 byte)
    _padding: [u8; 7],                  // +0x31: alignment padding (7 bytes)
}

impl SkBitmap {
    fn new(pixels: *const u8, width: i32, height: i32, row_bytes: usize) -> Self {
        Self {
            pixel_ref: std::ptr::null(),
            pixmap: SkPixmap {
                pixels,
                row_bytes,
                info: SkImageInfo {
                    color_space: std::ptr::null(),
                    color_type: SK_COLOR_TYPE_BGRA_8888,
                    alpha_type: SK_ALPHA_TYPE_OPAQUE,
                    width,
                    height,
                },
            },
            flags: 0,
            _padding: [0; 7],
        }
    }
}

// =============================================================================
// DLL function types
// =============================================================================

type GetLibraryVersionFn = unsafe extern "C" fn(*mut u32, *mut u32);
type SetFileContentFunctionsFn = unsafe extern "C" fn(
    get_size: extern "C" fn(*const i8) -> u32,
    get_content: extern "C" fn(*const i8, u32, *mut i8),
);
type InitOcrFn = unsafe extern "C" fn() -> bool;
type GetMaxImageDimensionFn = unsafe extern "C" fn() -> u32;
type PerformOcrFn = unsafe extern "C" fn(*const SkBitmap, *mut u32) -> *mut i8;
type FreeCharArrayFn = unsafe extern "C" fn(*mut i8);

// =============================================================================
// Global state for file callbacks
// =============================================================================

static mut MODEL_DIR: Option<String> = None;

extern "C" fn get_file_content_size(path: *const i8) -> u32 {
    unsafe {
        let model_dir = match &MODEL_DIR {
            Some(dir) => dir,
            None => return 0,
        };
        let path_str = match std::ffi::CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return 0,
        };
        let full_path = format!("{}\\{}", model_dir, path_str);
        match std::fs::metadata(&full_path) {
            Ok(meta) => meta.len() as u32,
            Err(_) => 0,
        }
    }
}

extern "C" fn get_file_content(path: *const i8, size: u32, buffer: *mut i8) {
    unsafe {
        let model_dir = match &MODEL_DIR {
            Some(dir) => dir,
            None => return,
        };
        let path_str = match std::ffi::CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return,
        };
        let full_path = format!("{}\\{}", model_dir, path_str);
        if let Ok(data) = std::fs::read(&full_path) {
            let copy_len = std::cmp::min(data.len(), size as usize);
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer as *mut u8, copy_len);
        }
    }
}

// =============================================================================
// NativeOCR implementation
// =============================================================================

pub struct NativeOCR {
    _library: Library,
    perform_ocr: PerformOcrFn,
    free_char_array: FreeCharArrayFn,
    pub max_dimension: u32,
    _hook: Option<native_hook::NativeHook>,
}

impl NativeOCR {
    pub fn new(model_dir: &Path) -> Result<Self> {
        Self::new_with_hook(model_dir, None)
    }

    pub fn new_with_hook(model_dir: &Path, hook_output_dir: Option<PathBuf>) -> Result<Self> {
        // Look for chrome_screen_ai.dll in multiple locations
        let chrome_dll_path = Self::find_dll(model_dir)?;

        // Set global model dir for callbacks
        unsafe {
            MODEL_DIR = Some(model_dir.to_string_lossy().to_string());
        }

        // Load DLL
        let library = unsafe { Library::new(&chrome_dll_path) }
            .map_err(|e| anyhow!("Failed to load chrome_screen_ai.dll: {}", e))?;

        unsafe {
            // Get function pointers
            let get_version: Symbol<GetLibraryVersionFn> = library.get(b"GetLibraryVersion\0")?;
            let set_file_funcs: Symbol<SetFileContentFunctionsFn> =
                library.get(b"SetFileContentFunctions\0")?;
            let init_ocr: Symbol<InitOcrFn> = library.get(b"InitOCRUsingCallback\0")?;
            let get_max_dim: Symbol<GetMaxImageDimensionFn> =
                library.get(b"GetMaxImageDimension\0")?;
            let perform_ocr: PerformOcrFn = *library.get(b"PerformOCR\0")?;
            let free_char_array: FreeCharArrayFn =
                *library.get(b"FreeLibraryAllocatedCharArray\0")?;

            // Get version
            let mut major = 0u32;
            let mut minor = 0u32;
            get_version(&mut major, &mut minor);
            println!("  NativeOCR: Chrome Screen AI v{}.{}", major, minor);

            // Set file callbacks
            set_file_funcs(get_file_content_size, get_file_content);

            // Initialize OCR
            if !init_ocr() {
                return Err(anyhow!("InitOCRUsingCallback failed"));
            }

            // Get max dimension
            let max_dimension = get_max_dim();
            println!("  NativeOCR: max dimension = {}", max_dimension);
            println!("  NativeOCR: initialized");

            // Install hook after DLL is loaded and initialized
            // (chrome_screen_ai.dll statically links TFLite and exports its C API)
            let hook = match hook_output_dir {
                Some(dir) => Some(native_hook::NativeHook::install(&chrome_dll_path, dir)?),
                None => None,
            };

            Ok(Self {
                _library: library,
                perform_ocr,
                free_char_array,
                max_dimension,
                _hook: hook,
            })
        }
    }

    /// Find chrome_screen_ai.dll in multiple locations
    fn find_dll(model_dir: &Path) -> Result<std::path::PathBuf> {
        // 1. Check in model directory (Chrome's layout)
        let dll_in_model = model_dir.join("chrome_screen_ai.dll");
        if dll_in_model.exists() {
            return Ok(dll_in_model);
        }

        // 2. Check relative to executable
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let dll_near_exe = exe_dir.join("chrome_screen_ai.dll");
                if dll_near_exe.exists() {
                    return Ok(dll_near_exe);
                }
            }
        }

        // 3. Check in current working directory
        let dll_in_cwd = std::path::PathBuf::from("chrome_screen_ai.dll");
        if dll_in_cwd.exists() {
            return Ok(dll_in_cwd);
        }

        Err(anyhow!(
            "chrome_screen_ai.dll not found. Checked:\n  - {}\n  - near executable\n  - current directory",
            dll_in_model.display()
        ))
    }

    /// Perform OCR on a grayscale image, returns raw protobuf bytes
    pub fn perform_ocr_raw(&self, image: &GrayImage) -> Result<Vec<u8>> {
        let (w, h) = (image.width() as i32, image.height() as i32);

        // Convert grayscale to BGRA
        let mut bgra_pixels = Vec::with_capacity((w * h * 4) as usize);
        for pixel in image.as_raw() {
            let v = *pixel;
            bgra_pixels.push(v); // B
            bgra_pixels.push(v); // G
            bgra_pixels.push(v); // R
            bgra_pixels.push(255); // A
        }

        let row_bytes = (w * 4) as usize;
        let bitmap = SkBitmap::new(bgra_pixels.as_ptr(), w, h, row_bytes);

        let mut result_len = 0u32;
        let result_ptr = unsafe { (self.perform_ocr)(&bitmap, &mut result_len) };

        // Flush any pending hook captures (data is valid now, after inference)
        if self._hook.is_some() {
            native_hook::flush_final_captures();
        }

        if result_ptr.is_null() {
            return Err(anyhow!("PerformOCR returned null"));
        }

        // Copy result
        let result = unsafe {
            std::slice::from_raw_parts(result_ptr as *const u8, result_len as usize).to_vec()
        };

        // Free DLL-allocated memory
        unsafe { (self.free_char_array)(result_ptr) };

        Ok(result)
    }

    /// Perform OCR and decode protobuf to extract text lines
    pub fn perform_ocr(&self, image: &GrayImage) -> Result<Vec<OcrLine>> {
        let raw = self.perform_ocr_raw(image)?;
        decode_ocr_result(&raw)
    }
}

// =============================================================================
// Protobuf decoding (manual, without schema)
// =============================================================================

/// Represents a recognized text line
#[derive(Debug, Clone)]
pub struct OcrLine {
    pub text: String,
    pub confidence: f32,
    pub bounding_box: Option<BoundingBox>,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

/// Simple protobuf wire type decoder
struct ProtobufReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ProtobufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn read_byte(&mut self) -> Option<u8> {
        if self.pos < self.data.len() {
            let b = self.data[self.pos];
            self.pos += 1;
            Some(b)
        } else {
            None
        }
    }

    fn read_varint(&mut self) -> Option<u64> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            let b = self.read_byte()?;
            result |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                return None;
            }
        }
        Some(result)
    }

    fn read_bytes(&mut self, len: usize) -> Option<&'a [u8]> {
        if self.pos + len <= self.data.len() {
            let slice = &self.data[self.pos..self.pos + len];
            self.pos += len;
            Some(slice)
        } else {
            None
        }
    }

    fn skip(&mut self, len: usize) {
        self.pos = std::cmp::min(self.pos + len, self.data.len());
    }

    fn read_tag(&mut self) -> Option<(u32, u32)> {
        let varint = self.read_varint()?;
        let field_num = (varint >> 3) as u32;
        let wire_type = (varint & 0x7) as u32;
        Some((field_num, wire_type))
    }

    fn read_f32(&mut self) -> Option<f32> {
        let bytes = self.read_bytes(4)?;
        Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

/// Decode OCR result protobuf to extract text lines
fn decode_ocr_result(data: &[u8]) -> Result<Vec<OcrLine>> {
    let mut lines = Vec::new();
    let mut reader = ProtobufReader::new(data);

    // The protobuf structure appears to be:
    // message OcrResult {
    //   repeated LineAnnotation lines = 2;
    // }
    // message LineAnnotation {
    //   repeated WordAnnotation words = 1;
    //   string text = 2;  // or could be in words
    //   float confidence = 3;
    //   BoundingBox box = ?;
    // }

    while reader.remaining() > 0 {
        let (field_num, wire_type) = match reader.read_tag() {
            Some(t) => t,
            None => break,
        };

        match wire_type {
            0 => {
                // Varint
                reader.read_varint();
            }
            1 => {
                // 64-bit
                reader.skip(8);
            }
            2 => {
                // Length-delimited
                let len = reader.read_varint().unwrap_or(0) as usize;
                if field_num == 2 {
                    // This is likely a LineAnnotation
                    let line_data = reader.read_bytes(len).unwrap_or(&[]);
                    if let Some(line) = decode_line_annotation(line_data) {
                        lines.push(line);
                    }
                } else {
                    reader.skip(len);
                }
            }
            5 => {
                // 32-bit
                reader.skip(4);
            }
            _ => break,
        }
    }

    Ok(lines)
}

/// Decode a single line annotation
fn decode_line_annotation(data: &[u8]) -> Option<OcrLine> {
    let mut reader = ProtobufReader::new(data);
    let mut text = String::new();
    let mut confidence = 0.0f32;
    let mut words_text = Vec::new();

    while reader.remaining() > 0 {
        let (field_num, wire_type) = reader.read_tag()?;

        match wire_type {
            0 => {
                // Varint
                reader.read_varint();
            }
            1 => {
                // 64-bit
                reader.skip(8);
            }
            2 => {
                // Length-delimited
                let len = reader.read_varint()? as usize;
                let bytes = reader.read_bytes(len)?;

                if field_num == 1 {
                    // Could be a WordAnnotation submessage
                    if let Some(word) = decode_word_annotation(bytes) {
                        words_text.push(word);
                    }
                } else if field_num == 2 || field_num == 3 {
                    // Could be text field - try to decode as UTF-8
                    if let Ok(s) = std::str::from_utf8(bytes) {
                        if !s.is_empty() && s.chars().all(|c| !c.is_control() || c == ' ') {
                            text = s.to_string();
                        }
                    }
                }
            }
            5 => {
                // 32-bit (float)
                if field_num == 3 || field_num == 4 {
                    confidence = reader.read_f32().unwrap_or(0.0);
                } else {
                    reader.skip(4);
                }
            }
            _ => break,
        }
    }

    // Use words if no direct text
    if text.is_empty() && !words_text.is_empty() {
        text = words_text.join("");
    }

    if text.is_empty() {
        return None;
    }

    Some(OcrLine {
        text,
        confidence,
        bounding_box: None,
    })
}

/// Decode a word annotation to extract text
fn decode_word_annotation(data: &[u8]) -> Option<String> {
    let mut reader = ProtobufReader::new(data);
    let mut text = String::new();

    while reader.remaining() > 0 {
        let (field_num, wire_type) = match reader.read_tag() {
            Some(t) => t,
            None => break,
        };

        match wire_type {
            0 => {
                reader.read_varint();
            }
            1 => {
                reader.skip(8);
            }
            2 => {
                let len = reader.read_varint().unwrap_or(0) as usize;
                let bytes = reader.read_bytes(len).unwrap_or(&[]);

                // Field 2 is often the text in word annotations
                if field_num == 2 {
                    if let Ok(s) = std::str::from_utf8(bytes) {
                        if !s.is_empty() {
                            text = s.to_string();
                        }
                    }
                }
            }
            5 => {
                reader.skip(4);
            }
            _ => break,
        }
    }

    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}
