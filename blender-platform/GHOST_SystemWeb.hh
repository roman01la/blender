/* SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once

#include "GHOST_System.hh"

#include <functional>
#include <memory>
#include <set>

class GHOST_WindowWeb;
class GHOST_WebGPUCanvas;

/** GHOST window and input on the Blender runtime worker. */
class GHOST_SystemWeb final : public GHOST_System {
 public:
  ~GHOST_SystemWeb() override;

  /**
   * Prepares the device before the synchronous GHOST window creation call.
   * The callback receives a null error on success. Disposal cancels notification.
   */
  void requestWebGPU(std::function<void(const char *error)> callback);
  bool isWebGPUReady() const;

  GHOST_IWindow *createWindow(const char *title,
                              int32_t left,
                              int32_t top,
                              uint32_t width,
                              uint32_t height,
                              GHOST_TWindowState state,
                              GHOST_GPUSettings gpu_settings,
                              bool exclusive,
                              bool is_dialog,
                              const GHOST_IWindow *parent_window) override;
  GHOST_TSuccess disposeWindow(GHOST_IWindow *window) override;
  GHOST_IContext *createOffscreenContext(GHOST_GPUSettings gpu_settings) override;
  GHOST_TSuccess disposeContext(GHOST_IContext *context) override;
  bool processEvents(bool wait_for_event) override;
  uint64_t getMilliSeconds() const override;
  uint8_t getNumDisplays() const override;
  void getMainDisplayDimensions(uint32_t &width, uint32_t &height) const override;
  void getAllDisplayDimensions(uint32_t &width, uint32_t &height) const override;
  GHOST_TCapabilityFlag getCapabilities() const override;
  GHOST_TSuccess getCursorPosition(int32_t &x, int32_t &y) const override;
  GHOST_TSuccess setCursorPosition(int32_t x, int32_t y) override;
  bool setConsoleWindowState(GHOST_TConsoleWindowState action) override;
  char *getClipboard(bool selection) const override;
  void putClipboard(const char *buffer, bool selection) const override;

  void resize(int width, int height, float pixel_ratio);
  void pointer(int x, int y, unsigned buttons, const GHOST_TabletData &tablet);
  void releaseButtons();
  void wheel(double x, double y);
  void scroll(double x, double y, bool magnify);
  void focus(bool active);
  void keyboard(bool down, const char *code, const char *text, unsigned modifiers, bool repeat);

 protected:
  GHOST_TSuccess getModifierKeys(GHOST_ModifierKeys &keys) const override;
  GHOST_TSuccess getButtons(GHOST_Buttons &buttons) const override;

 private:
  friend class GHOST_WindowWeb;
  struct WebGPUInitialization;

  GHOST_WebGPUCanvas *webGPUCanvas() const;
  void updateButtons(unsigned buttons, const GHOST_TabletData &tablet);

  GHOST_WindowWeb *window_ = nullptr;
  std::shared_ptr<WebGPUInitialization> webgpu_;
  std::set<GHOST_TKey> pressed_keys_;
  unsigned buttons_ = 0;
  int32_t cursor_x_ = 0;
  int32_t cursor_y_ = 0;
  bool focused_ = false;
  bool has_cursor_ = false;
  double scroll_x_ = 0;
  double scroll_y_ = 0;
  double precise_scroll_x_ = 0;
  double precise_scroll_y_ = 0;
  double magnify_scroll_ = 0;
  GHOST_TabletData tablet_ = GHOST_TABLET_DATA_NONE;
};
