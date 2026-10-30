"""Простая TTS-говорилка для Windows на движке Piper.

Озвучивает текст локально (без интернета) голосами из загружаемых
ONNX-моделей. Модели можно скачать прямо из приложения или добавить вручную.
"""

import shutil
import threading
import tkinter as tk
import urllib.request
import wave
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd
import soundfile as sf
from piper import PiperVoice

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# Голоса для скачивания (HuggingFace, репозиторий rhasspy/piper-voices)
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
DOWNLOADABLE_VOICES = {
    "Дмитрий (RU, мужской)": "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium",
    "Ирина (RU, женский)": "ru/ru_RU/irina/medium/ru_RU-irina-medium",
    "Руслан (RU, мужской)": "ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium",
    "Amy (EN, женский)": "en/en_US/amy/medium/en_US-amy-medium",
    "Libritts R (EN, мужской)": "en/en_US/libritts_r/medium/en_US-libritts_r-medium",
}


class TalkerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Говорилка (Piper TTS)")
        self.geometry("560x480")

        self._voice = None  # загруженная PiperVoice
        self._voice_path = None
        self._busy = False
        self._playing = False

        self._build_ui()
        self._refresh_models()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- Интерфейс ----------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        model_frame = ttk.LabelFrame(self, text="Голосовая модель")
        model_frame.pack(fill="x", **pad)

        row1 = ttk.Frame(model_frame)
        row1.pack(fill="x", padx=8, pady=5)
        self.model_combo = ttk.Combobox(row1, state="readonly", width=38)
        self.model_combo.pack(side="left")
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._load_selected_model())
        ttk.Button(row1, text="↻", width=3, command=self._refresh_models).pack(side="left", padx=4)

        row2 = ttk.Frame(model_frame)
        row2.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(row2, text="Загрузить из файла…", command=self._load_model_from_file).pack(side="left")
        ttk.Button(row2, text="Скачать голос…", command=self._download_dialog).pack(side="left", padx=5)
        ttk.Button(row2, text="Открыть папку моделей", command=self._open_models_folder).pack(side="left")

        text_frame = ttk.LabelFrame(self, text="Текст")
        text_frame.pack(fill="both", expand=True, **pad)
        self.text_box = tk.Text(text_frame, wrap="word", height=8, font=("Segoe UI", 11))
        self.text_box.pack(fill="both", expand=True, padx=8, pady=8)
        self.text_box.insert("1.0", "Привет! Введите текст, и я его озвучу.")

        controls = ttk.Frame(self)
        controls.pack(fill="x", **pad)
        self.speak_btn = ttk.Button(controls, text="🔊 Сказать", command=self._speak)
        self.speak_btn.pack(side="left")
        ttk.Button(controls, text="⏹ Стоп", command=self._stop_playback).pack(side="left", padx=5)
        ttk.Button(controls, text="💾 Сохранить в WAV", command=self._save_wav).pack(side="left")

        ttk.Label(controls, text="Скорость:").pack(side="left", padx=(15, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed = ttk.Scale(controls, from_=0.5, to=2.0, variable=self.speed_var, length=120)
        speed.pack(side="left")

        self.status_label = ttk.Label(self, text="Выберите модель", relief="sunken", anchor="w")
        self.status_label.pack(fill="x", side="bottom")

    # ---------- Модели ----------

    def _find_models(self):
        if not MODELS_DIR.exists():
            return []
        return sorted(MODELS_DIR.rglob("*.onnx"))

    def _refresh_models(self):
        models = self._find_models()
        self.model_combo["values"] = [m.name for m in models]
        self._model_paths = {m.name: m for m in models}
        if models:
            self.model_combo.current(0)
            self._load_selected_model()
        else:
            self.status_label.configure(
                text="Моделей нет — нажмите «Скачать голос…» или «Загрузить из файла…»"
            )

    def _load_selected_model(self):
        name = self.model_combo.get()
        path = self._model_paths.get(name)
        if path is None:
            return
        try:
            self._voice = PiperVoice.load(str(path))
            self._voice_path = path
            self.status_label.configure(text=f"Модель загружена: {name}")
        except Exception as exc:  # noqa: BLE001
            self._voice = None
            messagebox.showerror("Ошибка загрузки модели", str(exc))

    def _load_model_from_file(self):
        """Копирует выбранную .onnx-модель (и её .json-конфиг) в папку models."""
        path = filedialog.askopenfilename(
            title="Выберите файл модели",
            filetypes=[("Piper-модели", "*.onnx"), ("Все файлы", "*.*")],
        )
        if not path:
            return
        src = Path(path)
        MODELS_DIR.mkdir(exist_ok=True)
        try:
            shutil.copy2(src, MODELS_DIR / src.name)
            config = src.with_suffix(src.suffix + ".json")
            if config.exists():
                shutil.copy2(config, MODELS_DIR / config.name)
            else:
                messagebox.showwarning(
                    "Нет конфига",
                    f"Рядом с моделью не найден файл {config.name}.\n"
                    "Без него модель может не загрузиться.",
                )
        except OSError as exc:
            messagebox.showerror("Ошибка копирования", str(exc))
            return
        self._refresh_models()
        self.model_combo.set(src.name)
        self._load_selected_model()

    def _open_models_folder(self):
        MODELS_DIR.mkdir(exist_ok=True)
        import os
        os.startfile(MODELS_DIR)  # noqa: S606

    # ---------- Скачивание голосов ---------

    def _download_dialog(self):
        dialog = tk.Toplevel(self)
        dialog.title("Скачать голос")
        dialog.geometry("340x260")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text="Выберите голос для скачивания:").pack(padx=10, pady=(10, 5))
        listbox = tk.Listbox(dialog)
        for name in DOWNLOADABLE_VOICES:
            listbox.insert(tk.END, name)
        listbox.pack(fill="both", expand=True, padx=10)
        listbox.selection_set(0)

        def start_download():
            sel = listbox.curselection()
            if not sel:
                return
            name = listbox.get(sel[0])
            dialog.destroy()
            self._download_voice(name)

        ttk.Button(dialog, text="Скачать", command=start_download).pack(pady=10)

    def _download_voice(self, name):
        if self._busy:
            return
        rel = DOWNLOADABLE_VOICES[name]
        base_name = rel.rsplit("/", 1)[-1]
        MODELS_DIR.mkdir(exist_ok=True)
        targets = [
            (f"{HF_BASE}/{rel}.onnx", MODELS_DIR / f"{base_name}.onnx"),
            (f"{HF_BASE}/{rel}.onnx.json", MODELS_DIR / f"{base_name}.onnx.json"),
        ]
        self._busy = True
        self.speak_btn.configure(state="disabled")

        def worker():
            try:
                for url, dest in targets:
                    self._set_status(f"Скачиваю {dest.name}…")
                    urllib.request.urlretrieve(url, dest)  # noqa: S310
                self._set_status(f"Голос «{name}» скачан")
                self.after(0, self._refresh_models)
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: messagebox.showerror("Ошибка скачивания", str(exc)))
                self._set_status("Ошибка скачивания")
            finally:
                self._busy = False
                self.after(0, lambda: self.speak_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    # ---------- Синтез и воспроизведение ----------

    def _set_status(self, text):
        self.after(0, lambda: self.status_label.configure(text=text))

    def _synthesize(self, text, wav_path):
        """Синтезирует текст в WAV-файл. Вызывать из фонового потока."""
        length_scale = 1.0 / max(0.1, self.speed_var.get())
        from piper import SynthesisConfig
        syn_config = SynthesisConfig(length_scale=length_scale)
        with wave.open(str(wav_path), "wb") as wav_file:
            first = True
            for chunk in self._voice.synthesize(text, syn_config=syn_config):
                if first:
                    wav_file.setnchannels(chunk.sample_channels)
                    wav_file.setsampwidth(chunk.sample_width)
                    wav_file.setframerate(chunk.sample_rate)
                    first = False
                wav_file.writeframes(chunk.audio_int16_bytes)

    def _get_text(self):
        return self.text_box.get("1.0", tk.END).strip()

    def _speak(self):
        if self._voice is None:
            messagebox.showwarning("Нет модели", "Сначала выберите или скачайте голосовую модель.")
            return
        text = self._get_text()
        if not text:
            messagebox.showinfo("Пусто", "Введите текст для озвучки.")
            return
        if self._busy:
            return
        self._stop_playback()
        self._busy = True
        self.speak_btn.configure(state="disabled")

        def worker():
            try:
                OUTPUT_DIR.mkdir(exist_ok=True)
                tmp = OUTPUT_DIR / "_last_speech.wav"
                self._set_status("Синтезирую…")
                self._synthesize(text, tmp)
                data, samplerate = sf.read(tmp, dtype="float32")
                self._playing = True
                sd.play(data, samplerate)
                self._set_status("Говорю…")
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda: messagebox.showerror("Ошибка синтеза", str(exc)))
                self._set_status("Ошибка синтеза")
            finally:
                self._busy = False
                self.after(0, lambda: self.speak_btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_playback(self):
        if self._playing:
            sd.stop()
            self._playing = False

    def _save_wav(self):
        if self._voice is None:
            messagebox.showwarning("Нет модели", "Сначала выберите или скачайте голосовую модель.")
            return
        text = self._get_text()
        if not text:
            messagebox.showinfo("Пусто", "Введите текст для озвучки.")
            return
        path = filedialog.asksaveasfilename(
            title="Сохранить озвучку",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
            initialfile="speech.wav",
        )
        if not path:
            return
        try:
            self._synthesize(text, Path(path))
            self.status_label.configure(text=f"Сохранено: {path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Ошибка синтеза", str(exc))

    def _on_close(self):
        self._stop_playback()
        self.destroy()


if __name__ == "__main__":
    app = TalkerApp()
    app.mainloop()
