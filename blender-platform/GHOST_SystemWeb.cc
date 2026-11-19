/* SPDX-License-Identifier: GPL-2.0-or-later */

#include "GHOST_SystemWeb.hh"

#include "GHOST_ContextNone.hh"
#include "GHOST_ContextWebGPU.hh"
#include "GHOST_EventButton.hh"
#include "GHOST_EventCursor.hh"
#include "GHOST_EventKey.hh"
#include "GHOST_EventTrackpad.hh"
#include "GHOST_EventWheel.hh"
#include "GHOST_TimerManager.hh"
#include "GHOST_WindowManager.hh"

#include <emscripten.h>
#include <emscripten/threading.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <string_view>
#include <utility>

namespace {
constexpr char canvas_selector[] = "#gpu";
GHOST_SystemWeb *attached_system = nullptr;
constexpr std::array browser_buttons = {GHOST_kButtonMaskLeft,
                                        GHOST_kButtonMaskRight,
                                        GHOST_kButtonMaskMiddle,
                                        GHOST_kButtonMaskButton4,
                                        GHOST_kButtonMaskButton5};

// clang-format off
EM_JS(bool, webgpu_canvas_available, (), {
  if (!globalThis.navigator?.gpu || !(Module.canvas instanceof OffscreenCanvas)) {
    return false;
  }
  GL.offscreenCanvases.gpu = Module.canvas;
  return Boolean(Module.canvas.getContext("webgpu"));
});

EM_JS(void, attach_canvas, (bool memory64), {
  Module.ghostBrowserWindow.attach({
    resize: _ghost_web_resize,
    pointer: _ghost_web_pointer,
    releaseButtons: _ghost_web_release_buttons,
    wheel: _ghost_web_wheel,
    scroll: _ghost_web_scroll,
    focus: _ghost_web_focus,
    key(down, code, key, modifiers, repeat) {
      const stack = stackSave();
      try {
        const nativeCode = stringToUTF8OnStack(code);
        const nativeKey = stringToUTF8OnStack(key);
        _ghost_web_key(down,
                       memory64 ? BigInt(nativeCode) : nativeCode,
                       memory64 ? BigInt(nativeKey) : nativeKey,
                       modifiers,
                       repeat);
      } finally {
        stackRestore(stack);
      }
    }
  });
});

EM_JS(void, detach_canvas, (), { Module.ghostBrowserWindow.detach(); });
EM_JS(void, resize_canvas, (unsigned width, unsigned height), {
  Module.ghostBrowserWindow.send("resize", width >>> 0, height >>> 0);
});
EM_JS(void, maximize_canvas, (bool maximize), {
  Module.ghostBrowserWindow.send("maximize", Boolean(maximize));
});
EM_JS(void, set_canvas_cursor, (const char *cursor), {
  Module.ghostBrowserWindow.send("cursor", UTF8ToString(Number(cursor)));
});
EM_JS(void, set_canvas_cursor_grab, (bool enabled), {
  Module.ghostBrowserWindow.send("cursorGrab", Boolean(enabled));
});
EM_JS(void, set_page_title, (const char *title), {
  Module.ghostBrowserWindow.send("title", UTF8ToString(Number(title)));
});
EM_JS(int, get_screen_dimension, (bool height), {
  return Module.ghostBrowserWindow.display[height ? 1 : 0];
});
// clang-format on

GHOST_TKey key_from_code(std::string_view code)
{
  if (code.starts_with("Key") && code.size() == 4 && code[3] >= 'A' && code[3] <= 'Z') {
    return GHOST_TKey(GHOST_kKeyA + code[3] - 'A');
  }
  if (code.starts_with("Digit") && code.size() == 6 && code[5] >= '0' && code[5] <= '9') {
    return GHOST_TKey(GHOST_kKey0 + code[5] - '0');
  }
  if (code.starts_with("Numpad") && code.size() == 7 && code[6] >= '0' && code[6] <= '9') {
    return GHOST_TKey(GHOST_kKeyNumpad0 + code[6] - '0');
  }
  if (code.starts_with("F") && code.size() >= 2 && code.size() <= 3 && code[1] >= '1' &&
      code[1] <= '9')
  {
    int number = code[1] - '0';
    if (code.size() == 3) {
      number = code[2] >= '0' && code[2] <= '9' ? number * 10 + code[2] - '0' : 0;
    }
    if (number >= 1 && number <= 24) {
      return GHOST_TKey(GHOST_kKeyF1 + number - 1);
    }
  }

  constexpr std::pair<std::string_view, GHOST_TKey> keys[] = {
      {"AltLeft", GHOST_kKeyLeftAlt},
      {"AltRight", GHOST_kKeyRightAlt},
      {"ArrowDown", GHOST_kKeyDownArrow},
      {"ArrowLeft", GHOST_kKeyLeftArrow},
      {"ArrowRight", GHOST_kKeyRightArrow},
      {"ArrowUp", GHOST_kKeyUpArrow},
      {"Backquote", GHOST_kKeyAccentGrave},
      {"Backslash", GHOST_kKeyBackslash},
      {"Backspace", GHOST_kKeyBackSpace},
      {"BracketLeft", GHOST_kKeyLeftBracket},
      {"BracketRight", GHOST_kKeyRightBracket},
      {"CapsLock", GHOST_kKeyCapsLock},
      {"Comma", GHOST_kKeyComma},
      {"ContextMenu", GHOST_kKeyApp},
      {"ControlLeft", GHOST_kKeyLeftControl},
      {"ControlRight", GHOST_kKeyRightControl},
      {"Delete", GHOST_kKeyDelete},
      {"End", GHOST_kKeyEnd},
      {"Enter", GHOST_kKeyEnter},
      {"Equal", GHOST_kKeyEqual},
      {"Escape", GHOST_kKeyEsc},
      {"Home", GHOST_kKeyHome},
      {"Insert", GHOST_kKeyInsert},
      {"IntlBackslash", GHOST_kKeyGrLess},
      {"MetaLeft", GHOST_kKeyLeftOS},
      {"MetaRight", GHOST_kKeyRightOS},
      {"Minus", GHOST_kKeyMinus},
      {"NumLock", GHOST_kKeyNumLock},
      {"NumpadAdd", GHOST_kKeyNumpadPlus},
      {"NumpadDecimal", GHOST_kKeyNumpadPeriod},
      {"NumpadDivide", GHOST_kKeyNumpadSlash},
      {"NumpadEnter", GHOST_kKeyNumpadEnter},
      {"NumpadMultiply", GHOST_kKeyNumpadAsterisk},
      {"NumpadSubtract", GHOST_kKeyNumpadMinus},
      {"PageDown", GHOST_kKeyDownPage},
      {"PageUp", GHOST_kKeyUpPage},
      {"Pause", GHOST_kKeyPause},
      {"Period", GHOST_kKeyPeriod},
      {"PrintScreen", GHOST_kKeyPrintScreen},
      {"Quote", GHOST_kKeyQuote},
      {"ScrollLock", GHOST_kKeyScrollLock},
      {"Semicolon", GHOST_kKeySemicolon},
      {"ShiftLeft", GHOST_kKeyLeftShift},
      {"ShiftRight", GHOST_kKeyRightShift},
      {"Slash", GHOST_kKeySlash},
      {"Space", GHOST_kKeySpace},
      {"Tab", GHOST_kKeyTab},
  };
  for (const auto &[name, key] : keys) {
    if (name == code) {
      return key;
    }
  }
  return GHOST_kKeyUnknown;
}

const char *css_cursor(GHOST_TStandardCursor shape)
{
  switch (shape) {
    case GHOST_kStandardCursorDefault:
    case GHOST_kStandardCursorLeftArrow:
    case GHOST_kStandardCursorRightArrow:
      return "default";
    case GHOST_kStandardCursorWait:
      return "wait";
    case GHOST_kStandardCursorText:
      return "text";
    case GHOST_kStandardCursorCrosshair:
      return "crosshair";
    case GHOST_kStandardCursorMove:
      return "move";
    case GHOST_kStandardCursorHandPoint:
      return "pointer";
    case GHOST_kStandardCursorHandOpen:
      return "grab";
    case GHOST_kStandardCursorHandClosed:
      return "grabbing";
    case GHOST_kStandardCursorLeftRight:
      return "ew-resize";
    case GHOST_kStandardCursorUpDown:
      return "ns-resize";
    case GHOST_kStandardCursorZoomIn:
      return "zoom-in";
    case GHOST_kStandardCursorZoomOut:
      return "zoom-out";
    default:
      return nullptr;
  }
}
}  // namespace

extern "C" {
EMSCRIPTEN_KEEPALIVE void ghost_web_resize(int width, int height, float ratio)
{
  attached_system->resize(width, height, ratio);
}
EMSCRIPTEN_KEEPALIVE void ghost_web_pointer(
    int x, int y, unsigned buttons, int tablet, float pressure, float tilt_x, float tilt_y)
{
  const GHOST_TabletData data = tablet ? GHOST_TabletData{GHOST_TTabletMode(tablet),
                                                          pressure,
                                                          tilt_x / 90.0f,
                                                          tilt_y / 90.0f} :
                                         GHOST_TABLET_DATA_NONE;
  attached_system->pointer(x, y, buttons, data);
}
EMSCRIPTEN_KEEPALIVE void ghost_web_release_buttons()
{
  attached_system->releaseButtons();
}
EMSCRIPTEN_KEEPALIVE void ghost_web_wheel(double x, double y)
{
  attached_system->wheel(x, y);
}
EMSCRIPTEN_KEEPALIVE void ghost_web_scroll(double x, double y, int magnify)
{
  attached_system->scroll(x, y, magnify != 0);
}
EMSCRIPTEN_KEEPALIVE void ghost_web_focus(int active)
{
  attached_system->focus(active != 0);
}
EMSCRIPTEN_KEEPALIVE void ghost_web_key(
    int down, const char *code, const char *key, unsigned modifiers, int repeat)
{
  attached_system->keyboard(down != 0, code, key, modifiers, repeat != 0);
}
}

class GHOST_WindowWeb final : public GHOST_Window {
 public:
  GHOST_WindowWeb(GHOST_SystemWeb &system, const char *title, const GHOST_ContextParams &params)
      : GHOST_Window(0, 0, GHOST_kWindowStateNormal, params, false), system_(system), title_(title)
  {
  }

  bool getValid() const override
  {
    return true;
  }
  void setTitle(const char *title) override
  {
    title_ = title;
    set_page_title(title);
  }
  std::string getTitle() const override
  {
    return title_;
  }
  void getWindowBounds(GHOST_Rect &bounds) const override
  {
    getClientBounds(bounds);
  }
  void getClientBounds(GHOST_Rect &bounds) const override
  {
    bounds = GHOST_Rect(0, 0, width_, height_);
  }
  GHOST_TSuccess setClientWidth(uint32_t width) override
  {
    return setClientSize(width, height_);
  }
  GHOST_TSuccess setClientHeight(uint32_t height) override
  {
    return setClientSize(width_, height);
  }
  GHOST_TSuccess setClientSize(uint32_t width, uint32_t height) override
  {
    if (state_ != GHOST_kWindowStateNormal) {
      return GHOST_kFailure;
    }
    resize_canvas(width, height);
    return GHOST_kSuccess;
  }
  void screenToClient(int32_t x, int32_t y, int32_t &out_x, int32_t &out_y) const override
  {
    out_x = x;
    out_y = y;
  }
  void clientToScreen(int32_t x, int32_t y, int32_t &out_x, int32_t &out_y) const override
  {
    screenToClient(x, y, out_x, out_y);
  }
  GHOST_TWindowState getState() const override
  {
    return state_;
  }
  GHOST_TSuccess setState(GHOST_TWindowState state) override
  {
    if (state != GHOST_kWindowStateNormal && state != GHOST_kWindowStateMaximized) {
      return GHOST_kFailure;
    }
    if (state != state_) {
      state_ = state;
      maximize_canvas(state_ == GHOST_kWindowStateMaximized);
    }
    return GHOST_kSuccess;
  }
  GHOST_TSuccess setOrder(GHOST_TWindowOrder /*order*/) override
  {
    return GHOST_kFailure;
  }
  GHOST_TSuccess invalidate() override
  {
    return system_.pushEvent(
        std::make_unique<GHOST_Event>(system_.getMilliSeconds(), GHOST_kEventWindowUpdate, this));
  }
  GHOST_TSuccess hasCursorShape(GHOST_TStandardCursor shape) override
  {
    return css_cursor(shape) ? GHOST_kSuccess : GHOST_kFailure;
  }
  bool resize(int width, int height, float scale)
  {
    /* A hidden canvas can have zero CSS size. Keep the last drawable until it returns. */
    if ((width <= 0 || height <= 0) && width_ > 0 && height_ > 0) {
      return false;
    }
    width = std::max(1, width);
    height = std::max(1, height);
    if (!std::isfinite(scale) || scale <= 0.0f) {
      scale = 1.0f;
    }
    if (const auto *canvas = system_.webGPUCanvas()) {
      /* Match the scale Blender uses for input and offscreen viewport textures. */
      scale = std::min(scale, float(canvas->max_texture_dimension()) / std::max(width, height));
    }
    if (width == width_ && height == height_ && scale == native_pixel_size_) {
      return false;
    }
    width_ = width;
    height_ = height;
    native_pixel_size_ = scale;
    if (getDrawingContextType() == GHOST_kDrawingContextTypeWebGPU) {
      resizeDrawingContext(*static_cast<GHOST_ContextWebGPU *>(getDrawingContext()));
    }
    return true;
  }

 private:
  GHOST_Context *newDrawingContext(GHOST_TDrawingContextType type) override
  {
    auto *canvas = system_.webGPUCanvas();
    if (type != GHOST_kDrawingContextTypeWebGPU || !canvas) {
      return nullptr;
    }
    auto context = std::make_unique<GHOST_ContextWebGPU>(want_context_params_, *canvas);
    if (!context->initializeDrawingContext()) {
      return nullptr;
    }
    resizeDrawingContext(*context);
    return context.release();
  }
  void resizeDrawingContext(GHOST_ContextWebGPU &context) const
  {
    const auto backing_size = [scale = native_pixel_size_](int logical) {
      return std::max(1u, uint32_t(float(logical) * scale));
    };
    context.resize(backing_size(width_), backing_size(height_));
  }
  GHOST_TSuccess setWindowCursorVisibility(bool visible) override
  {
    const char *cursor = css_cursor(getCursorShape());
    set_canvas_cursor(visible ? (cursor ? cursor : "default") : "none");
    return GHOST_kSuccess;
  }
  GHOST_TSuccess setWindowCursorGrab(GHOST_TGrabCursorMode mode) override
  {
    set_canvas_cursor_grab(mode == GHOST_kGrabWrap || mode == GHOST_kGrabHide);
    return GHOST_kSuccess;
  }
  GHOST_TSuccess setWindowCursorShape(GHOST_TStandardCursor shape) override
  {
    const char *cursor = css_cursor(shape);
    if (!cursor) {
      return GHOST_kFailure;
    }
    if (getCursorVisibility()) {
      set_canvas_cursor(cursor);
    }
    return GHOST_kSuccess;
  }
  GHOST_TSuccess setWindowCustomCursorShape(
      const uint8_t *, const uint8_t *, const int[2], const int[2], bool) override
  {
    return GHOST_kFailure;
  }

  GHOST_SystemWeb &system_;
  std::string title_;
  GHOST_TWindowState state_ = GHOST_kWindowStateNormal;
  int width_ = -1;
  int height_ = -1;
};

struct GHOST_SystemWeb::WebGPUInitialization {
  std::unique_ptr<GHOST_WebGPUCanvas> canvas;
  bool pending = false;
};

GHOST_SystemWeb::~GHOST_SystemWeb()
{
  if (window_) {
    disposeWindow(window_);
  }
}

GHOST_WebGPUCanvas *GHOST_SystemWeb::webGPUCanvas() const
{
  return webgpu_ ? webgpu_->canvas.get() : nullptr;
}

bool GHOST_SystemWeb::isWebGPUReady() const
{
  return webGPUCanvas() != nullptr;
}

void GHOST_SystemWeb::requestWebGPU(std::function<void(const char *)> callback)
{
  if (!emscripten_is_main_runtime_thread() || !webgpu_canvas_available()) {
    callback("WebGPU or the browser GHOST canvas is unavailable.");
    return;
  }
  if (webGPUCanvas()) {
    callback(nullptr);
    return;
  }
  if (!webgpu_) {
    webgpu_ = std::make_shared<WebGPUInitialization>();
  }
  if (webgpu_->pending) {
    callback("A WebGPU device request is already pending.");
    return;
  }
  webgpu_->pending = true;
  GHOST_WebGPUCanvas::request(canvas_selector,
                              [state = std::weak_ptr(webgpu_), callback = std::move(callback)](
                                  std::unique_ptr<GHOST_WebGPUCanvas> canvas, const char *error) {
                                if (auto request = state.lock()) {
                                  request->canvas = std::move(canvas);
                                  request->pending = false;
                                  callback(error);
                                }
                              });
}

GHOST_IWindow *GHOST_SystemWeb::createWindow(const char *title,
                                             int32_t /*left*/,
                                             int32_t /*top*/,
                                             uint32_t /*width*/,
                                             uint32_t /*height*/,
                                             GHOST_TWindowState state,
                                             GHOST_GPUSettings gpu_settings,
                                             bool /*exclusive*/,
                                             bool is_dialog,
                                             const GHOST_IWindow *parent_window)
{
  const bool context_available = gpu_settings.context_type == GHOST_kDrawingContextTypeNone ||
                                 (gpu_settings.context_type == GHOST_kDrawingContextTypeWebGPU &&
                                  webGPUCanvas());
  if (!emscripten_is_main_runtime_thread() || attached_system || is_dialog || parent_window ||
      (state != GHOST_kWindowStateNormal && state != GHOST_kWindowStateMaximized) ||
      !context_available)
  {
    return nullptr;
  }
  window_ = new GHOST_WindowWeb(
      *this, title, GHOST_CONTEXT_PARAMS_FROM_GPU_SETTINGS(gpu_settings));
  window_manager_->addWindow(window_);
  attached_system = this;
  attach_canvas(sizeof(void *) == 8);
  window_->setState(state);
  if (!window_->setDrawingContextType(gpu_settings.context_type)) {
    disposeWindow(window_);
    return nullptr;
  }
  window_->setTitle(title);
  return window_;
}

GHOST_TSuccess GHOST_SystemWeb::disposeWindow(GHOST_IWindow *window)
{
  if (!window_ || window != window_) {
    return GHOST_kFailure;
  }
  window_->setState(GHOST_kWindowStateNormal);
  detach_canvas();
  attached_system = nullptr;
  window_ = nullptr;
  pressed_keys_.clear();
  buttons_ = 0;
  focused_ = false;
  has_cursor_ = false;
  scroll_x_ = scroll_y_ = 0.0;
  precise_scroll_x_ = precise_scroll_y_ = magnify_scroll_ = 0.0;
  tablet_ = GHOST_TABLET_DATA_NONE;
  return GHOST_System::disposeWindow(window);
}

GHOST_IContext *GHOST_SystemWeb::createOffscreenContext(GHOST_GPUSettings gpu_settings)
{
  const GHOST_ContextParams params = GHOST_CONTEXT_PARAMS_FROM_GPU_SETTINGS_OFFSCREEN(
      gpu_settings);
  if (gpu_settings.context_type == GHOST_kDrawingContextTypeNone) {
    return new GHOST_ContextNone(params);
  }
  auto *canvas = webGPUCanvas();
  if (!emscripten_is_main_runtime_thread() ||
      gpu_settings.context_type != GHOST_kDrawingContextTypeWebGPU || !canvas)
  {
    return nullptr;
  }
  auto context = std::make_unique<GHOST_ContextWebGPU>(params, canvas->device());
  if (!context->initializeDrawingContext()) {
    return nullptr;
  }
  return context.release();
}

GHOST_TSuccess GHOST_SystemWeb::disposeContext(GHOST_IContext *context)
{
  delete context;
  return GHOST_kSuccess;
}

bool GHOST_SystemWeb::processEvents(bool /*wait_for_event*/)
{
  const bool fired_timer = getTimerManager()->fireTimers(getMilliSeconds());
  return fired_timer || event_manager_->getNumEvents() != 0;
}

uint64_t GHOST_SystemWeb::getMilliSeconds() const
{
  return uint64_t(emscripten_get_now());
}

uint8_t GHOST_SystemWeb::getNumDisplays() const
{
  return 1;
}

void GHOST_SystemWeb::getMainDisplayDimensions(uint32_t &width, uint32_t &height) const
{
  width = get_screen_dimension(false);
  height = get_screen_dimension(true);
}

void GHOST_SystemWeb::getAllDisplayDimensions(uint32_t &width, uint32_t &height) const
{
  getMainDisplayDimensions(width, height);
}

GHOST_TCapabilityFlag GHOST_SystemWeb::getCapabilities() const
{
  /* The browser owns the window frame. Skip Blender's client-side decorations. */
  return GHOST_kCapabilityWindowDecorationServerSide;
}

GHOST_TSuccess GHOST_SystemWeb::getCursorPosition(int32_t &x, int32_t &y) const
{
  x = cursor_x_;
  y = cursor_y_;
  return has_cursor_ ? GHOST_kSuccess : GHOST_kFailure;
}

GHOST_TSuccess GHOST_SystemWeb::setCursorPosition(int32_t /*x*/, int32_t /*y*/)
{
  return GHOST_kFailure;
}
bool GHOST_SystemWeb::setConsoleWindowState(GHOST_TConsoleWindowState /*action*/)
{
  return false;
}
char *GHOST_SystemWeb::getClipboard(bool /*selection*/) const
{
  return nullptr;
}
void GHOST_SystemWeb::putClipboard(const char * /*buffer*/, bool /*selection*/) const {}

void GHOST_SystemWeb::resize(int width, int height, float pixel_ratio)
{
  if (window_->resize(width, height, pixel_ratio)) {
    pushEvent(std::make_unique<GHOST_Event>(getMilliSeconds(), GHOST_kEventWindowSize, window_));
  }
}

void GHOST_SystemWeb::pointer(int x, int y, unsigned buttons, const GHOST_TabletData &tablet)
{
  if (!has_cursor_ || x != cursor_x_ || y != cursor_y_ || tablet.Active != tablet_.Active ||
      tablet.Pressure != tablet_.Pressure || tablet.Xtilt != tablet_.Xtilt ||
      tablet.Ytilt != tablet_.Ytilt)
  {
    cursor_x_ = x;
    cursor_y_ = y;
    tablet_ = tablet;
    has_cursor_ = true;
    pushEvent(std::make_unique<GHOST_EventCursor>(
        getMilliSeconds(), GHOST_kEventCursorMove, window_, x, y, tablet));
  }
  updateButtons(buttons, tablet);
}

void GHOST_SystemWeb::updateButtons(unsigned buttons, const GHOST_TabletData &tablet)
{
  const unsigned changed = buttons_ ^ buttons;
  buttons_ = buttons;
  for (size_t i = 0; i < browser_buttons.size(); ++i) {
    const unsigned bit = 1u << i;
    if (changed & bit) {
      const auto type = (buttons & bit) ? GHOST_kEventButtonDown : GHOST_kEventButtonUp;
      pushEvent(std::make_unique<GHOST_EventButton>(
          getMilliSeconds(), type, window_, browser_buttons[i], tablet));
    }
  }
}

void GHOST_SystemWeb::releaseButtons()
{
  updateButtons(0, tablet_);
}

static int take_scroll_delta(double &pending, double delta)
{
  if (!std::isfinite(delta)) {
    return 0;
  }
  pending += delta;
  const double whole = std::trunc(pending);
  pending -= whole;
  return int(std::clamp(whole, -1000.0, 1000.0));
}

void GHOST_SystemWeb::wheel(double x, double y)
{
  const auto push_ticks = [this](double &pending, double delta, GHOST_TEventWheelAxis axis) {
    const int ticks = take_scroll_delta(pending, delta);
    if (ticks) {
      pushEvent(std::make_unique<GHOST_EventWheel>(getMilliSeconds(), window_, axis, ticks));
    }
  };
  push_ticks(scroll_x_, x, GHOST_kEventWheelAxisHorizontal);
  push_ticks(scroll_y_, -y, GHOST_kEventWheelAxisVertical);
}

void GHOST_SystemWeb::scroll(double x, double y, bool magnify)
{
  const double scale = window_->getNativePixelSize();
  /* Blender's trackpad events use native pixels and an upwards-positive y delta. */
  const int dx = magnify ? take_scroll_delta(magnify_scroll_, -y * scale) :
                           take_scroll_delta(precise_scroll_x_, -x * scale);
  const int dy = magnify ? 0 : take_scroll_delta(precise_scroll_y_, -y * scale);
  if (dx || dy) {
    pushEvent(std::make_unique<GHOST_EventTrackpad>(
        getMilliSeconds(),
        window_,
        magnify ? GHOST_kTrackpadEventMagnify : GHOST_kTrackpadEventScroll,
        cursor_x_,
        cursor_y_,
        dx,
        dy,
        true));
  }
}

void GHOST_SystemWeb::focus(bool active)
{
  if (active == focused_) {
    return;
  }
  focused_ = active;
  if (active) {
    window_manager_->setActiveWindow(window_);
  }
  else {
    releaseButtons();
    for (const auto key : pressed_keys_) {
      pushEvent(std::make_unique<GHOST_EventKey>(
          getMilliSeconds(), GHOST_kEventKeyUp, window_, key, false));
    }
    pressed_keys_.clear();
    scroll_x_ = scroll_y_ = 0.0;
    precise_scroll_x_ = precise_scroll_y_ = magnify_scroll_ = 0.0;
    window_manager_->setWindowInactive(window_);
  }
  pushEvent(std::make_unique<GHOST_Event>(getMilliSeconds(),
                                          active ? GHOST_kEventWindowActivate :
                                                   GHOST_kEventWindowDeactivate,
                                          window_));
}

GHOST_TSuccess GHOST_SystemWeb::getModifierKeys(GHOST_ModifierKeys &keys) const
{
  for (int i = 0; i < GHOST_kModifierKeyNum; ++i) {
    const auto modifier = GHOST_TModifierKey(i);
    keys.set(modifier, pressed_keys_.contains(GHOST_ModifierKeys::getModifierKeyCode(modifier)));
  }
  return GHOST_kSuccess;
}

GHOST_TSuccess GHOST_SystemWeb::getButtons(GHOST_Buttons &buttons) const
{
  buttons.clear();
  for (size_t i = 0; i < browser_buttons.size(); ++i) {
    buttons.set(browser_buttons[i], (buttons_ & (1u << i)) != 0);
  }
  return GHOST_kSuccess;
}

void GHOST_SystemWeb::keyboard(
    bool down, const char *code, const char *text, unsigned modifiers, bool repeat)
{
  if (!focused_) {
    return;
  }
  const auto key = key_from_code(code);
  char utf8[sizeof(GHOST_TEventKeyData::utf8_buf)] = {};
  const auto first = static_cast<unsigned char>(text[0]);
  const size_t count = first < 0x80 ? 1 : first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;
  const bool control = modifiers & 1u;
  const bool alt = modifiers & 2u;
  const bool meta = modifiers & 4u;
  if (down && !meta && (!control || alt) && first >= 0x20 && std::strlen(text) == count) {
    std::memcpy(utf8, text, count);
  }
  if (key == GHOST_kKeyUnknown && utf8[0] == '\0') {
    return;
  }
  if (key != GHOST_kKeyUnknown) {
    if (down) {
      pressed_keys_.insert(key);
    }
    else {
      pressed_keys_.erase(key);
    }
  }
  pushEvent(std::make_unique<GHOST_EventKey>(getMilliSeconds(),
                                             down ? GHOST_kEventKeyDown : GHOST_kEventKeyUp,
                                             window_,
                                             key,
                                             down && repeat,
                                             utf8));
}
