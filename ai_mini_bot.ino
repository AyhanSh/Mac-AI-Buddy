/*
  AI Mini Bot — face + camera + pan/tilt head + speaker + mic

  Board:   Seeed XIAO ESP32S3 Sense
  Display: 1.5" SH1107 128x128 OLED, I2C 0x3C (SSD1306 128x64 selectable)
  Camera:  onboard OV2640 (Sense daughterboard)
  Servos:  via PCA9685 @ 0x40, CH0 = pan, CH2 = tilt
  Speaker: MAX98357 I2S amp, 4 ohm 3 W enclosed speaker
  Mic:     MAX4466 analog electret on ADC1

  Wiring as built:
    D0  GPIO1   -> MAX4466 OUT          (ADC1 — works with WiFi up)
    D1  GPIO2   -> spare
    D2  GPIO3   -> MAX98357 BCLK
    D3  GPIO4   -> MAX98357 LRC
    D4  GPIO5   -> SDA rail -> OLED, PCA9685
    D5  GPIO6   -> SCL rail -> OLED, PCA9685
    D6  GPIO43  -> MAX98357 DIN
    D7  GPIO44  -> spare
    D8-D10      -> reserved (Sense microSD)
    3V3         -> OLED VCC, PCA9685 VCC, MAX4466 VCC
    GND         -> star point, everything
    5V 4A brick -> PCA9685 V+/GND screw terminal; MAX98357 VIN/GND twisted in
    MAX98357 SD, GAIN -> open (enabled, 9 dB). Speaker across + and -.

  NOTE: no decoupling capacitors fitted. Mitigations are in software —
  100 kHz I2C, reduced OLED contrast, reduced WiFi TX power, conservative
  audio amplitude, servo motion deferred during camera capture.

  Board settings:
    Board:              XIAO_ESP32S3
    PSRAM:              OPI PSRAM      (required — audio clips live there)
    USB CDC On Boot:    Enabled
    esp32 core:         3.x

  Libraries:
    Adafruit GFX, Adafruit SH110X (or SSD1306), Adafruit PWM Servo Driver

  Endpoints — head / face / camera:
    GET  /                       control page
    GET  /capture                single JPEG frame
    GET  /look?pan=90&tilt=90    set head angles
    GET  /look/center
    GET  /look/status
    GET  /status                 full health JSON
    GET  /set?e=<emotion>        set face
    GET  /<emotion>              same, path form

  Endpoints — audio:
    POST /say                    body = raw int16 LE mono PCM @ 16000 Hz,
                                 Content-Type: application/octet-stream.
                                 Plays through the speaker, face -> TALKING.
    GET  /mic?ms=12000&silence=1000&thresh=120&lead=3000
                                 records from MAX4466, returns WAV. ms is the
                                 cap (max 15000). With silence>0 the recording
                                 endpoints itself: it stops once you have been
                                 quiet that long, or after `lead` if you never
                                 started. Omit silence for a fixed-length clip.
                                 face -> LISTENING while recording.
    GET  /level                  current mic RMS, JSON, for VAD on the Mac
    GET  /beep?f=880&ms=150      local tone
    GET  /volume?v=10            0-15 (10 = unity, above 10 is software gain)
    GET  /stop                   flush the speech queue
*/

// ============================================================
// CONFIG
// ============================================================

#define DRIVER_SSD1306  0
#define DRIVER_SH1107   1
#define DISPLAY_DRIVER  DRIVER_SH1107
#define OLED_ROTATION   0
#define OLED_CONTRAST   80

#define ENABLE_CAMERA   1
#define ENABLE_WEB_PAGE 1
#define ENABLE_AUDIO    1     // speaker
#define ENABLE_MIC      1     // MAX4466

// ---------- servos ----------
#define SERVO_CH_PAN    3
#define SERVO_CH_TILT   4
#define PAN_MIN         15
#define PAN_MAX         165
#define TILT_MIN        30
#define TILT_MAX        150
#define PAN_CENTER      90
#define TILT_CENTER     90
#define SERVO_US_MIN    500
#define SERVO_US_MAX    2500
#define SERVO_DEG_PER_SEC 84  // head-move speed; 30% slower than a 120 deg/s baseline
#define SERVO_STEP_MS     15  // ramp update interval

// ---------- speaker: MAX98357 on I2S1 ----------
#define I2S_BCLK_PIN    3     // D2
#define I2S_LRC_PIN     4     // D3
#define I2S_DOUT_PIN    43    // D6
#define AUDIO_RATE      16000 // Mac must send PCM at this rate
#define AUDIO_VOLUME    13    // 0-15; 10 = unity, above 10 is software gain
#define AUDIO_VOLUME_MAX 15   // clip-protected gain ceiling (15 = +3.5 dB)
#define SAY_MAX_BYTES   (2 * 1024 * 1024)   // ~65 s at 16 kHz mono

// ---------- mic: MAX4466 on ADC1 ----------
#define MIC_PIN         1     // D0
#define MIC_RATE        16000
#define MIC_FRAME       256   // VAD window, 16 ms at 16 kHz
#define MIC_MAX_MS      15000 // cap on one /mic call. The sampling loop blocks
                              // the core, so it hands a tick back to the idle
                              // task every ~256 ms to keep the 5 s watchdog fed.
#define MIC_SILENCE_MS  1000  // endpointing: stop this long after you stop talking
#define MIC_VAD_THRESH  120   // frame RMS in raw ADC counts that counts as
                              // speech — the Mac measures the room via /level
                              // and overrides this per turn
#define MIC_LEAD_MS     3000  // give up if nobody starts talking within this

const char* WIFI_SSID = "hauz";
const char* WIFI_PASS = "mehdi_bu_bu_bu_1";
#define WIFI_TIMEOUT_MS 20000

// ============================================================
// INCLUDES
// ============================================================

#include <Wire.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Adafruit_GFX.h>
#include <Adafruit_PWMServoDriver.h>
#include "esp_camera.h"
#include <esp_heap_caps.h>
#include <string.h>
#include <math.h>

#if ENABLE_AUDIO
  #include <driver/i2s.h>
#endif

#if ENABLE_WEB_PAGE
  #include "control_page.h"
#endif

// ============================================================
// EMOTION TYPE — must stay directly after the includes.
// ============================================================
// Arduino injects generated prototypes after the last preprocessor block near
// the top of the file. parseEmotion() returns this type, so it has to exist
// before that point or the build fails with "'Emotion' does not name a type".

enum Emotion : uint8_t {
  NEUTRAL, HAPPY, SAD, ANGRY, SURPRISED, THINKING, LISTENING, TALKING, SLEEP,
  SEARCHING, LOADING, SCANNING, WIFI_FACE, MEMORY, SAVING
};
Emotion current = NEUTRAL;

// Same reason: playClip() takes a Clip&, so the generated prototype would land
// above the definition if this lived down in the audio section.
// Declared unconditionally — an unused struct costs nothing.
struct Clip { int16_t* data; size_t samples; };

// ============================================================
// DISPLAY
// ============================================================

#define SCREEN_WIDTH 128

#if DISPLAY_DRIVER == DRIVER_SSD1306
  #include <Adafruit_SSD1306.h>
  #define SCREEN_HEIGHT 64
  Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
  #define COLOR_ON   SSD1306_WHITE
  #define COLOR_OFF  SSD1306_BLACK
#else
  #include <Adafruit_SH110X.h>
  #define SCREEN_HEIGHT 128
  Adafruit_SH1107 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1, 400000, 100000);
  #define COLOR_ON   SH110X_WHITE
  #define COLOR_OFF  SH110X_BLACK
#endif

bool displayBegin(uint8_t addr) {
#if DISPLAY_DRIVER == DRIVER_SSD1306
  return display.begin(SSD1306_SWITCHCAPVCC, addr);
#else
  return display.begin(addr, true);
#endif
}

const int CX = SCREEN_WIDTH / 2;
const int CY = SCREEN_HEIGHT / 2;
const int EYE_W   = (SCREEN_HEIGHT >= 128) ? 44 : 34;
const int EYE_H   = (SCREEN_HEIGHT >= 128) ? 60 : 40;
const int EYE_R   = (SCREEN_HEIGHT >= 128) ? 14 : 10;
const int EYE_GAP = (SCREEN_HEIGHT >= 128) ? 24 : 20;

// ============================================================
// STATE
// ============================================================

WebServer server(80);

bool cameraReady = false;
bool oledReady   = false;
bool servosReady = false;
bool wifiReady   = false;
bool audioReady  = false;
bool micReady    = false;
uint8_t oledAddr = 0;

int currentPan  = PAN_CENTER;
int currentTilt = TILT_CENTER;

volatile bool capturing = false;
volatile bool speaking  = false;
volatile bool listening = false;

#define PCA9685_ADDR 0x40
Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(PCA9685_ADDR, Wire);

unsigned long lastFrame = 0;
const int FRAME_MS = 33;

bool blinking = false;
unsigned long blinkStart = 0;
unsigned long nextBlink = 0;
const int BLINK_MS = 180;

// ============================================================
// CAMERA PINS (XIAO ESP32S3 Sense)
// ============================================================

#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   10
#define SIOD_GPIO_NUM   40
#define SIOC_GPIO_NUM   39
#define Y9_GPIO_NUM     48
#define Y8_GPIO_NUM     11
#define Y7_GPIO_NUM     12
#define Y6_GPIO_NUM     14
#define Y5_GPIO_NUM     16
#define Y4_GPIO_NUM     18
#define Y3_GPIO_NUM     17
#define Y2_GPIO_NUM     15
#define VSYNC_GPIO_NUM  38
#define HREF_GPIO_NUM   47
#define PCLK_GPIO_NUM   13

// ============================================================
// AUDIO OUT — MAX98357 on I2S1
// ============================================================
#if ENABLE_AUDIO

#define I2S_PORT I2S_NUM_1

QueueHandle_t clipQueue = NULL;
int audioVolume = AUDIO_VOLUME;

void* psAlloc(size_t n) {
  void* p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!p) p = malloc(n);
  return p;
}
void* psRealloc(void* p, size_t n) {
  void* q = heap_caps_realloc(p, n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!q) q = realloc(p, n);
  return q;
}

bool initAudio() {
  i2s_config_t cfg = {};
  cfg.mode                 = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX);
  cfg.sample_rate          = AUDIO_RATE;
  cfg.bits_per_sample      = I2S_BITS_PER_SAMPLE_16BIT;
  cfg.channel_format       = I2S_CHANNEL_FMT_RIGHT_LEFT;
  cfg.communication_format = I2S_COMM_FORMAT_STAND_I2S;
  cfg.intr_alloc_flags     = 0;
  cfg.dma_buf_count        = 8;
  cfg.dma_buf_len          = 256;
  cfg.use_apll             = false;
  cfg.tx_desc_auto_clear   = true;

  if (i2s_driver_install(I2S_PORT, &cfg, 0, NULL) != ESP_OK) {
    Serial.println("Audio: i2s_driver_install failed");
    return false;
  }
  i2s_pin_config_t p = {};
  p.mck_io_num   = I2S_PIN_NO_CHANGE;     // MAX98357 needs no MCLK
  p.bck_io_num   = I2S_BCLK_PIN;
  p.ws_io_num    = I2S_LRC_PIN;
  p.data_out_num = I2S_DOUT_PIN;
  p.data_in_num  = I2S_PIN_NO_CHANGE;
  if (i2s_set_pin(I2S_PORT, &p) != ESP_OK) {
    Serial.println("Audio: i2s_set_pin failed");
    return false;
  }
  i2s_zero_dma_buffer(I2S_PORT);
  Serial.println("Audio: ok");
  return true;
}

// Plays mono int16 -> stereo duplicate, volume scaled.
// Fades are conditional: fade in only when starting from silence, fade out
// only when nothing follows. Otherwise chunked speech warbles at every seam.
void playClip(const Clip& c, bool fadeIn, bool fadeOut) {
  const int N = 256;
  static int16_t out[N * 2];
  size_t written;
  size_t fade = AUDIO_RATE / 200;
  if (fade * 2 > c.samples) fade = c.samples / 4;

  for (size_t i = 0; i < c.samples; i += N) {
    size_t n = c.samples - i;
    if (n > (size_t)N) n = N;
    for (size_t k = 0; k < n; k++) {
      size_t idx = i + k;
      int32_t s = (int32_t)c.data[idx] * audioVolume / 10;
      if (fade) {
        if (fadeIn && idx < fade)
          s = s * (int32_t)idx / (int32_t)fade;
        else if (fadeOut && idx > c.samples - fade)
          s = s * (int32_t)(c.samples - idx) / (int32_t)fade;
      }
      // Saturate: above unity gain the sample can exceed int16 range;
      // clamp so it hard-clips cleanly instead of wrapping into noise.
      if (s >  32767) s =  32767;
      if (s < -32768) s = -32768;
      out[2 * k]     = (int16_t)s;
      out[2 * k + 1] = (int16_t)s;
    }
    i2s_write(I2S_PORT, out, n * 4, &written, portMAX_DELAY);
  }
}

void flushSilence() {
  static int16_t z[512] = {0};
  size_t w;
  for (int k = 0; k < 3; k++) i2s_write(I2S_PORT, z, sizeof(z), &w, portMAX_DELAY);
}

void audioTask(void*) {
  Clip c;
  bool wasPlaying = false;
  for (;;) {
    if (xQueueReceive(clipQueue, &c, portMAX_DELAY) != pdTRUE) continue;
    speaking = true;
    if (c.data && c.samples) {
      // A chunk that arrives while more are queued gets no fades at all,
      // so a stream of TTS chunks plays as one continuous utterance.
      bool last = (uxQueueMessagesWaiting(clipQueue) == 0);
      playClip(c, !wasPlaying, last);
      wasPlaying = !last;
    }
    free(c.data);
    if (uxQueueMessagesWaiting(clipQueue) == 0) {
      flushSilence();
      speaking = false;
      wasPlaying = false;
    }
  }
}

bool enqueueClip(int16_t* data, size_t samples) {
  if (!audioReady || !clipQueue) { free(data); return false; }
  Clip c = { data, samples };
  if (xQueueSend(clipQueue, &c, pdMS_TO_TICKS(50)) != pdTRUE) { free(data); return false; }
  return true;
}

void queueTone(uint16_t freq, uint16_t ms) {
  size_t n = (size_t)AUDIO_RATE * ms / 1000;
  if (!n) return;
  int16_t* d = (int16_t*)psAlloc(n * 2);
  if (!d) return;
  float ph = 0, step = 2.0f * PI * freq / AUDIO_RATE;
  for (size_t i = 0; i < n; i++) {
    d[i] = (freq > 0) ? (int16_t)(12000 * sinf(ph)) : 0;
    ph += step; if (ph > 2.0f * PI) ph -= 2.0f * PI;
  }
  enqueueClip(d, n);
}

void stopSpeech() {
  if (!clipQueue) return;
  Clip c;
  while (xQueueReceive(clipQueue, &c, 0) == pdTRUE) free(c.data);
}

// /say raw body receiver — grows a PSRAM buffer, hands it to the audio task.
uint8_t* sayBuf = NULL;
size_t   sayCap = 0, sayLen = 0;
bool     sayOk  = false;

void handleSayRaw() {
  HTTPRaw& r = server.raw();
  if (r.status == RAW_START) {
    free(sayBuf);
    sayCap = 128 * 1024; sayLen = 0; sayOk = true;
    sayBuf = (uint8_t*)psAlloc(sayCap);
    if (!sayBuf) { sayOk = false; sayCap = 0; }
    speaking = true;   // start the mouth moving while bytes arrive
  } else if (r.status == RAW_WRITE) {
    if (!sayOk) return;
    if (sayLen + r.currentSize > sayCap) {
      size_t nc = sayCap * 2;
      while (nc < sayLen + r.currentSize) nc *= 2;
      if (nc > SAY_MAX_BYTES) { sayOk = false; return; }
      uint8_t* nb = (uint8_t*)psRealloc(sayBuf, nc);
      if (!nb) { sayOk = false; return; }
      sayBuf = nb; sayCap = nc;
    }
    memcpy(sayBuf + sayLen, r.buf, r.currentSize);
    sayLen += r.currentSize;
  } else if (r.status == RAW_END) {
    if (sayOk && sayLen >= 2) {
      enqueueClip((int16_t*)sayBuf, sayLen / 2);
    } else {
      free(sayBuf);
      if (uxQueueMessagesWaiting(clipQueue) == 0) speaking = false;
    }
    sayBuf = NULL; sayCap = 0;
  } else {  // RAW_ABORTED
    free(sayBuf); sayBuf = NULL; sayCap = 0; sayOk = false;
    if (uxQueueMessagesWaiting(clipQueue) == 0) speaking = false;
  }
}

void handleSayDone() {
  if (!audioReady) { server.send(503, "text/plain", "audio not ready"); return; }
  if (!sayOk)      { server.send(413, "text/plain", "clip rejected or too large"); return; }
  server.send(200, "application/json",
              "{\"ok\":true,\"bytes\":" + String(sayLen) +
              ",\"seconds\":" + String((float)sayLen / 2 / AUDIO_RATE, 2) + "}");
}

void handleBeep() {
  int f  = server.hasArg("f")  ? server.arg("f").toInt()  : 880;
  int ms = server.hasArg("ms") ? server.arg("ms").toInt() : 150;
  queueTone(constrain(f, 0, 8000), constrain(ms, 10, 3000));
  server.send(200, "text/plain", "ok");
}

void handleVolume() {
  if (server.hasArg("v")) audioVolume = constrain(server.arg("v").toInt(), 0, AUDIO_VOLUME_MAX);
  server.send(200, "text/plain", "vol:" + String(audioVolume));
}

void handleStop() {
  stopSpeech();
  server.send(200, "text/plain", "ok");
}

#endif  // ENABLE_AUDIO

// ============================================================
// MIC — MAX4466 on ADC1
// ============================================================
#if ENABLE_MIC

int micBias = 2048;

bool initMic() {
  analogReadResolution(12);
  analogSetPinAttenuation(MIC_PIN, ADC_11db);
  int64_t sum = 0;
  for (int i = 0; i < 1000; i++) { sum += analogRead(MIC_PIN); delayMicroseconds(100); }
  micBias = sum / 1000;
  Serial.printf("Mic: bias %d (expect ~2048)\n", micBias);
  // A bias pinned at either rail means the module is unpowered or OUT is open.
  return micBias > 300 && micBias < 3800;
}

// Records into buf, returns samples taken and the measured effective rate.
//
// With silenceMs > 0 the recording endpoints itself: it waits up to leadMs for
// speech to start, then keeps going until silenceMs of continuous quiet. This
// has to happen here rather than on the Mac — the loop below owns the core for
// the whole recording, so nothing can reach in and stop it partway.
//
// thresh is the per-frame RMS in raw ADC counts that counts as speech. The Mac
// measures the actual room through /level and passes its own figure, because
// the right value depends on the MAX4466 pot and how noisy the desk is.
// silenceMs == 0 records the full span, which is what a plain /mic?ms= does.
size_t recordMic(int16_t* buf, size_t maxSamples, uint32_t* effRate,
                 uint32_t silenceMs, int thresh, uint32_t leadMs) {
  listening = true;
  const uint32_t frameMs = MIC_FRAME * 1000UL / MIC_RATE;
  uint32_t period = 1000000UL / MIC_RATE;
  uint32_t t0 = micros(), next = t0;
  int64_t sum = 0;
  size_t n = 0;
  bool speech = false;
  uint32_t quietMs = 0, elapsedMs = 0, frames = 0;

  while (n < maxSamples) {
    size_t take = maxSamples - n;
    if (take > MIC_FRAME) take = MIC_FRAME;
    size_t base = n;
    int64_t fsum = 0;
    for (size_t i = 0; i < take; i++) {
      while ((int32_t)(micros() - next) < 0) { }
      next += period;
      int v = analogRead(MIC_PIN);
      buf[n++] = (int16_t)v;
      sum  += v;
      fsum += v;
    }
    elapsedMs += frameMs;

    if (silenceMs) {
      // RMS about this frame's own mean, so a drifting bias cannot read as loud.
      int32_t fdc = (int32_t)(fsum / (int64_t)take);
      int64_t fsq = 0;
      for (size_t i = base; i < n; i++) {
        int32_t d = (int32_t)buf[i] - fdc;
        fsq += (int64_t)d * d;
      }
      int32_t rms = (int32_t)sqrt((double)fsq / (double)take);
      if (rms > thresh)             { speech = true; quietMs = 0; }
      else if (speech)              { quietMs += frameMs;
                                      if (quietMs >= silenceMs) break; }
      else if (elapsedMs >= leadMs) break;   // nobody started talking
    }

    // Hand a tick back so core 1's idle task runs and the 5 s task watchdog
    // stays fed — this is what lets a recording outlast the old 4 s cap.
    // Costs ~1 ms of audio every ~256 ms; resetting `next` afterwards makes
    // that a clean seam rather than a burst of catch-up samples, and the
    // effective rate written into the WAV header absorbs the difference.
    if ((++frames & 15) == 0) { vTaskDelay(1); next = micros(); }
  }
  uint32_t elapsed = micros() - t0;
  *effRate = elapsed ? (uint32_t)((uint64_t)n * 1000000ULL / elapsed) : MIC_RATE;

  // Remove DC and scale 12-bit to 16-bit
  int32_t dc = sum / (int64_t)n;
  for (size_t i = 0; i < n; i++) {
    int32_t s = ((int32_t)buf[i] - dc) * 16;
    buf[i] = (int16_t)constrain(s, -32767, 32767);
  }
  listening = false;
  return n;
}

void writeWavHeader(uint8_t* h, uint32_t dataBytes, uint32_t rate) {
  uint32_t byteRate = rate * 2;
  memcpy(h, "RIFF", 4);
  uint32_t riffSize = 36 + dataBytes;   memcpy(h + 4,  &riffSize, 4);
  memcpy(h + 8, "WAVEfmt ", 8);
  uint32_t fmtLen = 16;                  memcpy(h + 16, &fmtLen, 4);
  uint16_t pcm = 1, ch = 1, bits = 16, blockAlign = 2;
  memcpy(h + 20, &pcm, 2);  memcpy(h + 22, &ch, 2);
  memcpy(h + 24, &rate, 4); memcpy(h + 28, &byteRate, 4);
  memcpy(h + 32, &blockAlign, 2); memcpy(h + 34, &bits, 2);
  memcpy(h + 36, "data", 4); memcpy(h + 40, &dataBytes, 4);
}

void handleMic() {
  if (!micReady) { server.send(503, "text/plain", "mic not ready"); return; }
  // ms is the cap. silence>0 turns on endpointing, and then the call returns
  // as soon as you stop talking rather than always running the full span.
  int ms      = server.hasArg("ms")      ? server.arg("ms").toInt()      : 3000;
  int silence = server.hasArg("silence") ? server.arg("silence").toInt() : 0;
  int thresh  = server.hasArg("thresh")  ? server.arg("thresh").toInt()  : MIC_VAD_THRESH;
  int lead    = server.hasArg("lead")    ? server.arg("lead").toInt()    : MIC_LEAD_MS;
  ms      = constrain(ms, 100, MIC_MAX_MS);
  silence = constrain(silence, 0, 5000);
  lead    = constrain(lead, 200, ms);
  size_t want = (size_t)MIC_RATE * ms / 1000;

  int16_t* buf = (int16_t*)heap_caps_malloc(want * 2, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!buf) buf = (int16_t*)malloc(want * 2);
  if (!buf) { server.send(500, "text/plain", "no memory"); return; }

  uint32_t effRate;
  size_t n = recordMic(buf, want, &effRate, (uint32_t)silence, thresh, (uint32_t)lead);
  Serial.printf("mic: %u samples (%.2f s), effective %u Hz\n",
                (unsigned)n, (float)n / MIC_RATE, (unsigned)effRate);

  uint8_t hdr[44];
  writeWavHeader(hdr, n * 2, effRate);   // real rate, so playback pitch is right

  server.setContentLength(44 + n * 2);
  server.send(200, "audio/wav", "");
  WiFiClient c = server.client();
  c.write(hdr, 44);
  const size_t CH = 4096;
  uint8_t* p = (uint8_t*)buf;
  for (size_t off = 0; off < n * 2; off += CH) {
    size_t len = n * 2 - off; if (len > CH) len = CH;
    c.write(p + off, len);
  }
  free(buf);
}

void handleLevel() {
  if (!micReady) { server.send(503, "text/plain", "mic not ready"); return; }
  const int N = 320;   // 20 ms
  int64_t sum = 0, sumSq = 0;
  int32_t peak = 0;
  int v[N];
  for (int i = 0; i < N; i++) { v[i] = analogRead(MIC_PIN); sum += v[i]; delayMicroseconds(50); }
  int32_t dc = sum / N;
  for (int i = 0; i < N; i++) {
    int32_t d = v[i] - dc;
    sumSq += (int64_t)d * d;
    if (abs(d) > peak) peak = abs(d);
  }
  int32_t rms = sqrt((double)sumSq / N);
  server.send(200, "application/json",
              "{\"rms\":" + String(rms) + ",\"peak\":" + String(peak) + "}");
}

#endif  // ENABLE_MIC

// ============================================================
// FACE DRAWING
// ============================================================

float blinkOpenness() {
  if (!blinking) return 1.0f;
  unsigned long e = millis() - blinkStart;
  if (e >= (unsigned long)BLINK_MS) { blinking = false; return 1.0f; }
  float t = (float)e / BLINK_MS;
  return fabsf(t - 0.5f) * 2.0f;
}

void drawOpenEyes(float hScale, int yOffset, int wDelta) {
  int h = (int)(EYE_H * hScale); if (h < 4) h = 4;
  int w  = EYE_W + wDelta;
  int y  = CY - h / 2 + yOffset;
  int lx = CX - EYE_GAP / 2 - w;
  int rx = CX + EYE_GAP / 2;
  display.fillRoundRect(lx, y, w, h, EYE_R, COLOR_ON);
  display.fillRoundRect(rx, y, w, h, EYE_R, COLOR_ON);
}

void drawEyesShift(int xs, int ys, float hScale) {
  int h = (int)(EYE_H * hScale); if (h < 4) h = 4;
  int w  = EYE_W;
  int y  = CY - h / 2 + ys;
  int lx = CX - EYE_GAP / 2 - w + xs;
  int rx = CX + EYE_GAP / 2 + xs;
  display.fillRoundRect(lx, y, w, h, EYE_R, COLOR_ON);
  display.fillRoundRect(rx, y, w, h, EYE_R, COLOR_ON);
}

void drawHappyEyes() {
  int r = EYE_W / 2, y = CY - 2;
  int lcx = CX - EYE_GAP / 2 - r, rcx = CX + EYE_GAP / 2 + r;
  display.fillCircle(lcx, y, r, COLOR_ON);
  display.fillCircle(rcx, y, r, COLOR_ON);
  display.fillRect(0, y, SCREEN_WIDTH, r + 3, COLOR_OFF);
}

void drawSurprisedEyes() {
  int r = EYE_W / 2 + 2;
  int lcx = CX - EYE_GAP / 2 - r + 3, rcx = CX + EYE_GAP / 2 + r - 3;
  display.fillCircle(lcx, CY, r, COLOR_ON);
  display.fillCircle(rcx, CY, r, COLOR_ON);
  display.fillCircle(lcx, CY, r / 2, COLOR_OFF);
  display.fillCircle(rcx, CY, r / 2, COLOR_OFF);
}

void drawAngryEyes() {
  int h = EYE_H, w = EYE_W, y = CY - h / 2 + 2;
  int lx = CX - EYE_GAP / 2 - w, rx = CX + EYE_GAP / 2;
  display.fillRoundRect(lx, y, w, h, EYE_R, COLOR_ON);
  display.fillRoundRect(rx, y, w, h, EYE_R, COLOR_ON);
  int cw = w - 8, ch = h / 2;
  display.fillTriangle(lx + w, y, lx + w - cw, y, lx + w, y + ch, COLOR_OFF);
  display.fillTriangle(rx,     y, rx + cw,     y, rx,     y + ch, COLOR_OFF);
}

void drawSadEyes() {
  int h = (EYE_H * 3) / 4, w = EYE_W, y = CY - h / 2 + 6;
  int lx = CX - EYE_GAP / 2 - w, rx = CX + EYE_GAP / 2;
  display.fillRoundRect(lx, y, w, h, EYE_R, COLOR_ON);
  display.fillRoundRect(rx, y, w, h, EYE_R, COLOR_ON);
  int cw = w - 8, ch = h / 2;
  display.fillTriangle(lx,     y, lx + cw,     y, lx,     y + ch, COLOR_OFF);
  display.fillTriangle(rx + w, y, rx + w - cw, y, rx + w, y + ch, COLOR_OFF);
}

void drawThinkingEyes() { drawEyesShift(6, -4, 0.65f); }

void drawSleepEyes() {
  int w = EYE_W, lx = CX - EYE_GAP / 2 - w, rx = CX + EYE_GAP / 2;
  display.fillRoundRect(lx, CY - 2, w, 5, 2, COLOR_ON);
  display.fillRoundRect(rx, CY - 2, w, 5, 2, COLOR_ON);
  display.setTextColor(COLOR_ON);
  display.setTextSize(1);
  display.setCursor(SCREEN_WIDTH - 30, 8);  display.print("z");
  display.setCursor(SCREEN_WIDTH - 22, 0);  display.print("z");
}

void drawSearching() { drawEyesShift((int)(14 * sin(millis() / 180.0)), 0, 0.8f); }

void drawLoading() {
  int r = 16, dots = 12, head = (int)(millis() / 80) % dots;
  for (int i = 0; i < dots; i++) {
    float a = (2 * PI * i) / dots;
    int x = CX + (int)(r * cos(a)), y = CY + (int)(r * sin(a));
    int dist = (i - head + dots) % dots;
    int rad = (dist == 0) ? 3 : (dist == 1 ? 2 : (dist == 2 ? 1 : 0));
    if (rad > 0) display.fillCircle(x, y, rad, COLOR_ON);
  }
}

void drawScanning() {
  drawEyesShift(0, 0, 0.55f);
  int x = (int)(CX + (SCREEN_WIDTH / 2 - 4) * sin(millis() / 300.0));
  display.drawFastVLine(x, 0, SCREEN_HEIGHT, COLOR_ON);
}

void drawWifiFace() {
  int by = SCREEN_HEIGHT - 18;
  display.fillCircle(CX, by, 3, COLOR_ON);
  int phase = (int)(millis() / 300) % 4;
  for (int k = 1; k <= 3; k++) {
    if (k > phase) continue;
    int rr = k * 11;
    for (int deg = 225; deg <= 315; deg += 5) {
      float a = deg * PI / 180.0;
      display.drawPixel(CX + (int)(rr * cos(a)), by + (int)(rr * sin(a)), COLOR_ON);
    }
  }
}

void drawMemory() {
  drawEyesShift(-6, -3, 0.8f);
  int phase = (int)(millis() / 250) % 3, bx = SCREEN_WIDTH - 30, by = 22;
  for (int i = 0; i < 3; i++)
    if (i <= phase) display.fillCircle(bx + i * 7, by - i * 7, (i == 2 ? 1 : 2), COLOR_ON);
}

void drawSaving() {
  drawEyesShift(0, -5, 0.7f);
  int w = 80, x = (SCREEN_WIDTH - w) / 2, y = SCREEN_HEIGHT - 12, h = 7;
  display.drawRoundRect(x, y, w, h, 2, COLOR_ON);
  int fill = (int)(millis() / 18) % (w - 2);
  display.fillRect(x + 1, y + 1, fill, h - 2, COLOR_ON);
}

void renderFace() {
  display.clearDisplay();

  // Audio overrides the resting faces so the bot looks alive while it talks
  // or listens. An explicit emotion (happy, angry, ...) is left alone.
  Emotion shown = current;
  bool restingFace = (shown == NEUTRAL || shown == LISTENING || shown == TALKING);
  if (restingFace) {
    if (speaking)       shown = TALKING;
    else if (listening) shown = LISTENING;
  }

  switch (shown) {
    case NEUTRAL:   drawOpenEyes(blinkOpenness(), 0, 0); break;
    case LISTENING: {
      float pulse = 1.0f + 0.06f * sin(millis() / 250.0);
      drawOpenEyes(blinkOpenness() * pulse, 0, 2);
      break;
    }
    case TALKING: {
      int bob = (int)(2 * sin(millis() / 90.0));
      drawOpenEyes(blinkOpenness(), bob, 0);
      break;
    }
    case HAPPY:     drawHappyEyes();     break;
    case SAD:       drawSadEyes();       break;
    case ANGRY:     drawAngryEyes();     break;
    case SURPRISED: drawSurprisedEyes(); break;
    case THINKING:  drawThinkingEyes();  break;
    case SLEEP:     drawSleepEyes();     break;
    case SEARCHING: drawSearching();     break;
    case LOADING:   drawLoading();       break;
    case SCANNING:  drawScanning();      break;
    case WIFI_FACE: drawWifiFace();      break;
    case MEMORY:    drawMemory();        break;
    case SAVING:    drawSaving();        break;
  }
  display.display();
}

void updateBlink() {
  if (current != NEUTRAL && current != LISTENING && current != TALKING) return;
  unsigned long now = millis();
  if (!blinking && now > nextBlink) {
    blinking = true;
    blinkStart = now;
    nextBlink = now + random(2500, 6000);
  }
}

// ============================================================
// I2C
// ============================================================

void scanI2C() {
  Serial.println("I2C scan (SDA=D4/GPIO5, SCL=D5/GPIO6):");
  int n = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() != 0) continue;
    Serial.printf("  found 0x%02X", addr);
    if (addr == 0x3C || addr == 0x3D) Serial.print("  (OLED)");
    if (addr == PCA9685_ADDR)         Serial.print("  (PCA9685)");
    if (addr == 0x70)                 Serial.print("  (PCA all-call, normal)");
    Serial.println();
    n++;
  }
  if (n == 0) Serial.println("  nothing found — check 3V3 / GND / SDA / SCL rails");
}

bool i2cPresent(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

// ============================================================
// CAMERA
// ============================================================

bool initCamera() {
  Serial.printf("PSRAM: %s\n", psramFound() ? "found" : "NOT found (Tools > PSRAM = OPI PSRAM)");
  delay(300);

  camera_config_t config;
  memset(&config, 0, sizeof(config));
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.grab_mode    = CAMERA_GRAB_WHEN_EMPTY;
#if defined(CAMERA_FB_IN_PSRAM)
  config.fb_location  = CAMERA_FB_IN_PSRAM;
#endif
  config.fb_count = psramFound() ? 2 : 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x\n", err);
    return false;
  }
  sensor_t* s = esp_camera_sensor_get();
  if (s) { s->set_vflip(s, 1); s->set_hmirror(s, 0); }
  Serial.println("Camera: ok");
  return true;
}

// ============================================================
// SERVOS
// ============================================================

int clampInt(int v, int lo, int hi) { return (v < lo) ? lo : (v > hi) ? hi : v; }

void writeServoAngle(uint8_t ch, int angle) {
  angle = clampInt(angle, 0, 180);
  int us = SERVO_US_MIN + (angle * (SERVO_US_MAX - SERVO_US_MIN)) / 180;
  uint16_t tick = (uint16_t)(((uint32_t)us * 4096UL) / 20000UL);
  tick = clampInt(tick, 1, 4095);
  pca.setPWM(ch, 0, tick);
}

bool initServos() {
  if (!i2cPresent(PCA9685_ADDR)) {
    Serial.println("PCA9685: no ACK at 0x40");
    return false;
  }
  pca.begin();
  pca.setOscillatorFrequency(27000000);
  pca.setPWMFreq(50);
  delay(20);

  currentPan  = PAN_CENTER;
  currentTilt = TILT_CENTER;
  writeServoAngle(SERVO_CH_PAN,  currentPan);
  writeServoAngle(SERVO_CH_TILT, currentTilt);
  delay(400);

  Serial.printf("Servo wiggle: pan CH%d, tilt CH%d\n", SERVO_CH_PAN, SERVO_CH_TILT);
  writeServoAngle(SERVO_CH_PAN, PAN_CENTER - 20);   delay(300);
  writeServoAngle(SERVO_CH_PAN, PAN_CENTER + 20);   delay(300);
  writeServoAngle(SERVO_CH_PAN, PAN_CENTER);        delay(300);
  writeServoAngle(SERVO_CH_TILT, TILT_CENTER - 20); delay(300);
  writeServoAngle(SERVO_CH_TILT, TILT_CENTER + 20); delay(300);
  writeServoAngle(SERVO_CH_TILT, TILT_CENTER);      delay(300);
  Serial.println("Servos: ready");
  return true;
}

// Ramps pan/tilt together at SERVO_DEG_PER_SEC rather than snapping to the
// target, so head moves are visibly smooth instead of a jerky jump.
void setHead(int pan, int tilt) {
  int targetPan  = clampInt(pan,  PAN_MIN,  PAN_MAX);
  int targetTilt = clampInt(tilt, TILT_MIN, TILT_MAX);
  if (!servosReady) { currentPan = targetPan; currentTilt = targetTilt; return; }

  unsigned long waited = 0;
  while (capturing && waited < 500) { delay(5); waited += 5; }

  int startPan  = currentPan;
  int startTilt = currentTilt;
  int maxDelta  = max(abs(targetPan - startPan), abs(targetTilt - startTilt));
  unsigned long durationMs = (unsigned long)((float)maxDelta * 1000.0f / SERVO_DEG_PER_SEC);
  unsigned long steps = durationMs / SERVO_STEP_MS;
  if (steps < 1) steps = 1;

  for (unsigned long s = 1; s <= steps; s++) {
    float t = (float)s / (float)steps;
    currentPan  = startPan  + (int)roundf((targetPan  - startPan)  * t);
    currentTilt = startTilt + (int)roundf((targetTilt - startTilt) * t);
    writeServoAngle(SERVO_CH_PAN,  currentPan);
    writeServoAngle(SERVO_CH_TILT, currentTilt);
    delay(SERVO_STEP_MS);
  }
  currentPan  = targetPan;
  currentTilt = targetTilt;
  writeServoAngle(SERVO_CH_PAN,  currentPan);
  writeServoAngle(SERVO_CH_TILT, currentTilt);
  Serial.printf("head -> pan %d, tilt %d\n", currentPan, currentTilt);
}

// ============================================================
// WEB HANDLERS
// ============================================================

void handleCapture() {
  if (!cameraReady) { server.send(503, "text/plain", "camera not ready"); return; }
  capturing = true;
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { capturing = false; server.send(500, "text/plain", "capture failed"); return; }
  server.setContentLength(fb->len);
  server.send(200, "image/jpeg", "");
  server.client().write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
  capturing = false;
}

void handleLook() {
  if (!servosReady) { server.send(503, "text/plain", "servos not ready"); return; }
  int pan  = server.hasArg("pan")  ? server.arg("pan").toInt()  : currentPan;
  int tilt = server.hasArg("tilt") ? server.arg("tilt").toInt() : currentTilt;
  setHead(pan, tilt);
  server.send(200, "text/plain", "ok:pan=" + String(currentPan) + ",tilt=" + String(currentTilt));
}

void handleLookCenter() {
  if (!servosReady) { server.send(503, "text/plain", "servos not ready"); return; }
  setHead(PAN_CENTER, TILT_CENTER);
  server.send(200, "text/plain", "ok:center");
}

void handleLookStatus() {
  server.send(200, "application/json",
    "{\"pan\":" + String(currentPan) + ",\"tilt\":" + String(currentTilt) +
    ",\"ready\":" + String(servosReady ? "true" : "false") + "}");
}

const char* emotionName(Emotion e) {
  static const char* names[] = { "neutral","happy","sad","angry","surprised","thinking",
    "listening","talking","sleep","searching","loading","scanning","wifi","memory","saving" };
  return names[(uint8_t)e < 15 ? (uint8_t)e : 0];
}

void handleStatus() {
  String j = "{";
  j += "\"camera\":"    + String(cameraReady ? "true" : "false");
  j += ",\"oled\":"     + String(oledReady   ? "true" : "false");
  j += ",\"servos\":"   + String(servosReady ? "true" : "false");
  j += ",\"audio\":"    + String(audioReady  ? "true" : "false");
  j += ",\"mic\":"      + String(micReady    ? "true" : "false");
  j += ",\"speaking\":" + String(speaking    ? "true" : "false");
  j += ",\"listening\":"+ String(listening   ? "true" : "false");
#if ENABLE_AUDIO
  j += ",\"volume\":"   + String(audioVolume);
#endif
  j += ",\"emotion\":\"" + String(emotionName(current)) + "\"";
  j += ",\"pan\":"      + String(currentPan);
  j += ",\"tilt\":"     + String(currentTilt);
  j += ",\"rssi\":"     + String(WiFi.RSSI());
  j += ",\"heap\":"     + String(ESP.getFreeHeap());
  j += ",\"psram\":"    + String(ESP.getFreePsram());
  j += "}";
  server.send(200, "application/json", j);
}

Emotion parseEmotion(String s) {
  s.toLowerCase();
  if (s == "happy")     return HAPPY;
  if (s == "sad")       return SAD;
  if (s == "angry")     return ANGRY;
  if (s == "surprised") return SURPRISED;
  if (s == "thinking")  return THINKING;
  if (s == "listening") return LISTENING;
  if (s == "talking")   return TALKING;
  if (s == "sleep")     return SLEEP;
  if (s == "searching") return SEARCHING;
  if (s == "loading")   return LOADING;
  if (s == "scanning")  return SCANNING;
  if (s == "wifi")      return WIFI_FACE;
  if (s == "memory")    return MEMORY;
  if (s == "saving")    return SAVING;
  return NEUTRAL;
}

void handleSet() {
  if (!server.hasArg("e")) { server.send(400, "text/plain", "missing e"); return; }
  current = parseEmotion(server.arg("e"));
  server.send(200, "text/plain", "ok:" + server.arg("e"));
}

void handleRoot() {
#if ENABLE_WEB_PAGE
  server.send_P(200, "text/html", CONTROL_PAGE);
#else
  server.send(200, "text/html",
    "<html><body style='font-family:sans-serif'><h2>AI Mini Bot</h2>"
    "<p><a href='/capture'>capture</a> | <a href='/look/center'>center</a> | "
    "<a href='/status'>status</a></p></body></html>");
#endif
}

void setupRoutes() {
  server.on("/",        handleRoot);
  server.on("/set",     handleSet);
  server.on("/status",  handleStatus);
  server.on("/capture", handleCapture);

  server.on("/look/center", HTTP_GET, handleLookCenter);
  server.on("/look/status", HTTP_GET, handleLookStatus);
  server.on("/look",        HTTP_GET, handleLook);

#if ENABLE_AUDIO
  // Raw body: fn runs after the body is complete, ufn receives the chunks.
  server.on("/say",    HTTP_POST, handleSayDone, handleSayRaw);
  server.on("/beep",   HTTP_GET,  handleBeep);
  server.on("/volume", HTTP_GET,  handleVolume);
  server.on("/stop",   HTTP_GET,  handleStop);
#endif
#if ENABLE_MIC
  server.on("/mic",    HTTP_GET,  handleMic);
  server.on("/level",  HTTP_GET,  handleLevel);
#endif

  auto reg = [](const char* p, Emotion e) {
    server.on(p, [e]() { current = e; server.send(200, "text/plain", "ok"); });
  };
  reg("/neutral", NEUTRAL);     reg("/happy", HAPPY);         reg("/sad", SAD);
  reg("/angry", ANGRY);         reg("/surprised", SURPRISED); reg("/thinking", THINKING);
  reg("/listening", LISTENING); reg("/talking", TALKING);     reg("/sleep", SLEEP);
  reg("/searching", SEARCHING); reg("/loading", LOADING);     reg("/scanning", SCANNING);
  reg("/wifi", WIFI_FACE);      reg("/memory", MEMORY);       reg("/saving", SAVING);
}

// ============================================================
// SETUP
// ============================================================

void showBootLine(const char* l1, const char* l2) {
  if (!oledReady) return;
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(COLOR_ON);
  display.setCursor(0, 0);
  display.println(l1);
  if (l2) display.println(l2);
  display.display();
}

void setup() {
  Serial.begin(115200);
  delay(800);
  Serial.println("\n=== AI Mini Bot ===");

#if ENABLE_CAMERA
  cameraReady = initCamera();          // camera BEFORE Wire
#endif

  Wire.begin(D4, D5);
  Wire.setClock(100000);
  scanI2C();

  // --- OLED ---
  for (uint8_t a : { (uint8_t)0x3C, (uint8_t)0x3D }) {
    if (!i2cPresent(a)) continue;
    if (displayBegin(a)) { oledReady = true; oledAddr = a; break; }
  }
  if (oledReady) {
#if DISPLAY_DRIVER == DRIVER_SH1107
    display.setRotation(OLED_ROTATION);
#endif
    display.setContrast(OLED_CONTRAST);
    showBootLine("booting...", NULL);
    Serial.printf("OLED: ok @ 0x%02X\n", oledAddr);
  } else {
    Serial.println("OLED: not found");
  }

  // --- Servos ---
  servosReady = initServos();

  // --- Audio out ---
#if ENABLE_AUDIO
  audioReady = initAudio();
  if (audioReady) {
    clipQueue = xQueueCreate(24, sizeof(Clip));   // deep: TTS arrives in chunks
    xTaskCreatePinnedToCore(audioTask, "audio", 6144, NULL, 1, NULL, 1);
  }
#endif

  // --- Mic ---
#if ENABLE_MIC
  micReady = initMic();
  Serial.println(micReady ? "Mic: ok" : "Mic: bias out of range — check MAX4466 power/OUT");
#endif

  showBootLine(servosReady ? "servos: ok" : "servos: FAIL", "connecting wifi");
  randomSeed(micros());

  // --- WiFi ---
  WiFi.mode(WIFI_STA);
  WiFi.setTxPower(WIFI_POWER_11dBm);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi");
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < WIFI_TIMEOUT_MS) {
    delay(300); Serial.print(".");
  }
  Serial.println();
  wifiReady = (WiFi.status() == WL_CONNECTED);

  if (wifiReady) {
    Serial.print("IP: "); Serial.println(WiFi.localIP());
    if (oledReady) {
      display.clearDisplay(); display.setCursor(0, 0);
      display.println("wifi ok"); display.println(WiFi.localIP().toString());
      display.display(); delay(1500);
    }
  } else {
    Serial.println("WiFi failed — offline mode");
    showBootLine("wifi FAILED", "offline mode");
    delay(1500);
  }

  Serial.printf("camera:%s oled:%s servos:%s wifi:%s audio:%s mic:%s\n",
                cameraReady ? "ok" : "no", oledReady ? "ok" : "no",
                servosReady ? "ok" : "no", wifiReady ? "ok" : "no",
                audioReady  ? "ok" : "no", micReady  ? "ok" : "no");

  setupRoutes();
  server.begin();
  nextBlink = millis() + 2000;

#if ENABLE_AUDIO
  if (audioReady) {
    if (wifiReady) { queueTone(523, 90); queueTone(659, 90); queueTone(784, 140); }
    else           { queueTone(392, 140); queueTone(294, 200); }
  }
#endif
}

// ============================================================
// LOOP
// ============================================================

void loop() {
  if (wifiReady) server.handleClient();

  unsigned long now = millis();
  if (oledReady && now - lastFrame >= (unsigned long)FRAME_MS) {
    lastFrame = now;
    updateBlink();
    renderFace();
  }
}
