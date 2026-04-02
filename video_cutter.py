"""
Auto Video Cutter for TikTok
Analisa vídeos, encontra os melhores momentos e corta automaticamente
para formato TikTok (9:16 vertical, 15-60 segundos).
"""

import os
import json
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from scipy.signal import find_peaks
import subprocess
import tempfile


class VideoCutter:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_audio_energy(self, video_path, chunk_duration=0.5):
        """Analisa a energia do áudio para encontrar momentos de alta intensidade."""
        video = VideoFileClip(video_path)
        audio = video.audio

        if audio is None:
            video.close()
            return [], video.duration

        sample_rate = audio.fps
        duration = video.duration

        # Extrair áudio como array
        audio_array = audio.to_soundarray()

        if len(audio_array.shape) > 1:
            audio_mono = np.mean(audio_array, axis=1)
        else:
            audio_mono = audio_array

        # Calcular energia por chunks
        chunk_samples = int(chunk_duration * sample_rate)
        energies = []

        for i in range(0, len(audio_mono), chunk_samples):
            chunk = audio_mono[i:i + chunk_samples]
            energy = np.sqrt(np.mean(chunk ** 2))
            energies.append(energy)

        video.close()
        return energies, duration

    def find_best_moments(self, energies, chunk_duration=0.5, clip_duration=30,
                          num_clips=5, min_gap=10):
        """Encontra os melhores momentos baseado na energia do áudio."""
        if not energies:
            return []

        energies = np.array(energies)

        # Suavizar o sinal de energia
        kernel_size = max(3, int(2 / chunk_duration))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(energies, kernel, mode='same')

        # Encontrar picos de energia
        min_distance = max(1, int(min_gap / chunk_duration))
        threshold = np.percentile(smoothed, 70)

        peaks, properties = find_peaks(
            smoothed,
            height=threshold,
            distance=min_distance,
            prominence=np.std(smoothed) * 0.3
        )

        if len(peaks) == 0:
            # Se não encontrar picos, pegar os momentos com mais energia
            window_size = int(clip_duration / chunk_duration)
            rolling_energy = np.convolve(energies, np.ones(window_size), mode='valid')
            peaks = np.argsort(rolling_energy)[-num_clips:]

        # Converter picos para timestamps
        moments = []
        for peak in peaks:
            timestamp = peak * chunk_duration
            # Centralizar o clip no pico
            start = max(0, timestamp - clip_duration / 2)
            moments.append({
                "start": round(start, 2),
                "end": round(start + clip_duration, 2),
                "energy": float(smoothed[peak]) if peak < len(smoothed) else 0
            })

        # Ordenar por energia (melhores primeiro)
        moments.sort(key=lambda x: x["energy"], reverse=True)

        # Remover sobreposições
        filtered = []
        for moment in moments:
            overlap = False
            for selected in filtered:
                if (moment["start"] < selected["end"] and
                        moment["end"] > selected["start"]):
                    overlap = True
                    break
            if not overlap:
                filtered.append(moment)
            if len(filtered) >= num_clips:
                break

        return filtered

    def detect_scene_changes(self, video_path, threshold=30.0):
        """Detecta mudanças de cena no vídeo."""
        try:
            from scenedetect import detect, ContentDetector
            scene_list = detect(video_path, ContentDetector(threshold=threshold))
            scenes = []
            for scene in scene_list:
                scenes.append({
                    "start": scene[0].get_seconds(),
                    "end": scene[1].get_seconds()
                })
            return scenes
        except Exception:
            return []

    def crop_to_vertical(self, clip):
        """Converte vídeo horizontal para vertical (9:16) cortando o centro."""
        w, h = clip.size

        # Se já é vertical, retorna como está
        if h > w:
            return clip

        # Calcular dimensões 9:16
        target_ratio = 9 / 16
        current_ratio = w / h

        if current_ratio > target_ratio:
            # Vídeo é mais largo — cortar laterais
            new_w = int(h * target_ratio)
            x_center = w // 2
            x1 = x_center - new_w // 2
            clip = clip.crop(x1=x1, x2=x1 + new_w)
        else:
            # Vídeo é mais alto — cortar topo e base
            new_h = int(w / target_ratio)
            y_center = h // 2
            y1 = y_center - new_h // 2
            clip = clip.crop(y1=y1, y2=y1 + new_h)

        return clip

    def cut_clip(self, video_path, start, end, output_path,
                 vertical=True, resolution=(1080, 1920)):
        """Corta um clip do vídeo e salva no formato TikTok."""
        video = VideoFileClip(video_path)

        # Ajustar end se passar da duração
        end = min(end, video.duration)
        start = max(0, start)

        if end <= start:
            video.close()
            return None

        clip = video.subclip(start, end)

        # Converter para vertical
        if vertical:
            clip = self.crop_to_vertical(clip)

        # Redimensionar para resolução do TikTok
        clip = clip.resize(resolution)

        # Salvar com configurações otimizadas para TikTok
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            bitrate="8000k",
            audio_bitrate="192k",
            fps=30,
            preset="medium",
            logger=None
        )

        clip.close()
        video.close()
        return output_path

    def process_video(self, video_path, clip_duration=30, num_clips=5,
                      vertical=True, resolution=(1080, 1920)):
        """Processa um vídeo completo — analisa e gera os melhores clips."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        clip_output_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(clip_output_dir, exist_ok=True)

        results = {
            "source": video_path,
            "clips": [],
            "status": "processing"
        }

        # 1. Analisar energia do áudio
        print(f"[1/4] Analisando áudio de: {video_path}")
        energies, duration = self.analyze_audio_energy(video_path)

        # 2. Detectar cenas (opcional)
        print("[2/4] Detectando mudanças de cena...")
        scenes = self.detect_scene_changes(video_path)

        # 3. Encontrar melhores momentos
        print("[3/4] Encontrando os melhores momentos...")
        clip_duration = min(clip_duration, 60)  # TikTok max 60s
        clip_duration = max(clip_duration, 15)  # Mínimo 15s

        if energies:
            moments = self.find_best_moments(
                energies,
                clip_duration=clip_duration,
                num_clips=num_clips
            )
        else:
            # Sem áudio — dividir vídeo em partes iguais
            moments = []
            step = duration / num_clips
            for i in range(num_clips):
                start = i * step
                moments.append({
                    "start": round(start, 2),
                    "end": round(start + clip_duration, 2),
                    "energy": 0
                })

        # 4. Cortar clips
        print(f"[4/4] Cortando {len(moments)} clips...")
        for i, moment in enumerate(moments):
            output_path = os.path.join(
                clip_output_dir,
                f"clip_{i + 1}__{moment['start']:.0f}s_a_{moment['end']:.0f}s.mp4"
            )
            print(f"  Cortando clip {i + 1}/{len(moments)}: "
                  f"{moment['start']:.1f}s → {moment['end']:.1f}s")

            result = self.cut_clip(
                video_path,
                moment["start"],
                moment["end"],
                output_path,
                vertical=vertical,
                resolution=resolution
            )

            if result:
                results["clips"].append({
                    "path": output_path,
                    "start": moment["start"],
                    "end": moment["end"],
                    "duration": round(moment["end"] - moment["start"], 2),
                    "energy_score": round(moment.get("energy", 0), 4)
                })

        results["status"] = "completed"
        results["total_clips"] = len(results["clips"])

        # Salvar relatório
        report_path = os.path.join(clip_output_dir, "report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nConcluído! {len(results['clips'])} clips salvos em: {clip_output_dir}")
        return results


# === USO DIRETO VIA TERMINAL ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto Video Cutter para TikTok")
    parser.add_argument("video", help="Caminho do vídeo para processar")
    parser.add_argument("--clips", type=int, default=5, help="Número de clips (padrão: 5)")
    parser.add_argument("--duracao", type=int, default=30, help="Duração de cada clip em segundos (padrão: 30)")
    parser.add_argument("--horizontal", action="store_true", help="Manter formato horizontal")
    parser.add_argument("--output", default="output", help="Pasta de saída (padrão: output)")

    args = parser.parse_args()

    cutter = VideoCutter(output_dir=args.output)
    results = cutter.process_video(
        args.video,
        clip_duration=args.duracao,
        num_clips=args.clips,
        vertical=not args.horizontal
    )

    print(f"\nRelatório salvo em: {os.path.join(args.output, 'report.json')}")
