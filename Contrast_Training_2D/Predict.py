import os
import sys
import argparse
from typing import List, Dict, Tuple

import numpy as np
import librosa

import torch
import torch.nn.functional as F


# 确保可以从当前目录导入 Train.py 中的定义
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
	sys.path.insert(0, SCRIPT_DIR)

from Train import AudioEncoder, DEFAULT_MUSIC_ROOT, _get_checkpoint_path


def load_checkpoint_2d(device: torch.device, data_root: str) -> Dict:
	ckpt_path = _get_checkpoint_path(data_root)
	if not os.path.isfile(ckpt_path):
		raise FileNotFoundError(f"[2D] 找不到 checkpoint 文件: {ckpt_path}")
	state = torch.load(ckpt_path, map_location=device)
	return state


def _preprocess_single_audio_2d(
	audio_path: str,
	sr: int,
	segment_duration: float,
	n_mels: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	if not os.path.isfile(audio_path):
		raise FileNotFoundError(f"音频文件不存在: {audio_path}")

	y, _ = librosa.load(audio_path, sr=sr, mono=True)
	target_len = int(segment_duration * sr)
	if len(y) >= target_len:
		total_len = len(y)
		start = max(0, (total_len - target_len) // 2)
		y = y[start : start + target_len]
	else:
		pad = target_len - len(y)
		y = np.pad(y, (0, pad), mode="constant")

	max_val = np.max(np.abs(y)) if len(y) > 0 else 0.0
	if max_val > 0:
		y = y / max_val

	harm, perc = librosa.effects.hpss(y)

	def _to_logmel(x: np.ndarray) -> torch.Tensor:
		S = librosa.feature.melspectrogram(
			y=x,
			sr=sr,
			n_mels=n_mels,
			fmin=20.0,
			fmax=sr / 2.0,
			n_fft=1024,
			hop_length=512,
			power=2.0,
		)
		S_db = librosa.power_to_db(S, ref=np.max)
		mu = float(np.mean(S_db))
		sigma = float(np.std(S_db)) + 1e-6
		S_norm = (S_db - mu) / sigma
		return torch.from_numpy(S_norm).unsqueeze(0).unsqueeze(0).float()

	feat_harm = _to_logmel(harm)
	feat_perc = _to_logmel(perc)
	feat_full = _to_logmel(y)
	return feat_perc, feat_harm, feat_full


def _build_confidence_report(score: torch.Tensor) -> Dict[str, float | str]:
	"""把相似度分数转换成可用于低置信提示的依据。"""
	probs = F.softmax(score, dim=0)
	top2_prob, _ = torch.topk(probs, k=min(2, probs.shape[0]))
	top2_score, _ = torch.topk(score, k=min(2, score.shape[0]))
	top1_prob = float(top2_prob[0].item())
	top2_prob_value = float(top2_prob[1].item()) if top2_prob.numel() > 1 else 0.0
	top1_score = float(top2_score[0].item())
	top2_score_value = float(top2_score[1].item()) if top2_score.numel() > 1 else top1_score
	prob_margin = top1_prob - top2_prob_value
	score_margin = top1_score - top2_score_value

	# 多类别下 top1_prob 常被类别数稀释，因此主要看 top1 相对 top2 的优势。
	pair_confidence = top1_prob / (top1_prob + top2_prob_value + 1e-8)
	margin_confidence = float(1.0 / (1.0 + np.exp(-8.0 * score_margin)))
	confidence = 0.7 * pair_confidence + 0.3 * margin_confidence

	if confidence >= 0.72:
		level = "high"
	elif confidence >= 0.62:
		level = "medium"
	else:
		level = "low"
	return {
		"top1_prob": top1_prob,
		"top2_prob": top2_prob_value,
		"prob_margin": prob_margin,
		"top1_score": top1_score,
		"top2_score": top2_score_value,
		"score_margin": score_margin,
		"pair_confidence": pair_confidence,
		"margin_confidence": margin_confidence,
		"confidence": confidence,
		"level": level,
	}


def predict_genre_2d(
	audio_path: str,
	data_root: str = DEFAULT_MUSIC_ROOT,
	alpha_drum: float | None = None,
) -> Tuple[str, List[Tuple[str, float]], Dict[str, float | str]]:
	"""使用 2D 模型对单首歌曲进行预测, 得到 Top-1 及所有风格融合得分。"""

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	state = load_checkpoint_2d(device, data_root)
	class_names: List[str] = list(state["class_names"])
	sr: int = int(state["sr"])
	n_mels: int = int(state["n_mels"])
	segment_duration: float = float(state["segment_duration"])
	embed_dim: int = int(state["embed_dim"])
	alpha_drum_global: float = float(state.get("alpha_drum", 0.5))
	alpha_drum_global = float(max(0.0, min(1.0, alpha_drum_global)))

	drum_encoder = AudioEncoder(n_mels=n_mels, embed_dim=embed_dim).to(device)
	timbre_encoder = AudioEncoder(n_mels=n_mels, embed_dim=embed_dim).to(device)
	drum_encoder.load_state_dict(state["drum_encoder_state_dict"])
	timbre_encoder.load_state_dict(state["timbre_encoder_state_dict"])
	drum_encoder.eval()
	timbre_encoder.eval()
	jazz_encoder = None
	prototype_jazz_1d = None
	jazz_style_idx = int(state.get("jazz_style_idx", -1))
	if "jazz_encoder_state_dict" in state and state.get("prototype_jazz_1d", None) is not None and 0 <= jazz_style_idx < len(class_names):
		jazz_encoder = AudioEncoder(n_mels=n_mels, embed_dim=embed_dim).to(device)
		jazz_encoder.load_state_dict(state["jazz_encoder_state_dict"])
		jazz_encoder.eval()
		prototype_jazz_1d = state["prototype_jazz_1d"].to(device)

	prototypes_drum: torch.Tensor = state["prototypes_drum"].to(device)
	prototypes_timbre: torch.Tensor = state["prototypes_timbre"].to(device)

	feat_perc, feat_harm, feat_full = _preprocess_single_audio_2d(
		audio_path=audio_path,
		sr=sr,
		segment_duration=segment_duration,
		n_mels=n_mels,
	)
	feat_perc = feat_perc.to(device)
	feat_harm = feat_harm.to(device)
	feat_full = feat_full.to(device)

	with torch.no_grad():
		emb_drum = F.normalize(drum_encoder(feat_perc), dim=1)
		emb_timbre = F.normalize(timbre_encoder(feat_harm), dim=1)
		sims_drum = torch.matmul(emb_drum, prototypes_drum.T).squeeze(0)
		sims_timbre = torch.matmul(emb_timbre, prototypes_timbre.T).squeeze(0)

	# 如果显式传入 alpha_drum, 或 checkpoint 中没有 per-class 权重, 则退回到全局权重融合
	alpha_drum_per_class: torch.Tensor | None = None
	if "alpha_drum_per_class" in state:
		alpha_drum_per_class = state["alpha_drum_per_class"].to(device).view(-1)
		if alpha_drum_per_class.shape[0] != len(class_names):
			alpha_drum_per_class = None

	if alpha_drum is not None or alpha_drum_per_class is None:
		alpha_eff = float(alpha_drum if alpha_drum is not None else alpha_drum_global)
		alpha_eff = float(max(0.0, min(1.0, alpha_eff)))
		alpha_timbre_eff = 1.0 - alpha_eff
		score = alpha_eff * sims_drum + alpha_timbre_eff * sims_timbre
	else:
		# 使用每个风格自己的鼓/音色权重进行融合
		alpha_vec = torch.clamp(alpha_drum_per_class, 0.0, 1.0)
		score = alpha_vec * sims_drum + (1.0 - alpha_vec) * sims_timbre

		# Jazz 类别使用 1D 分支得分替代 2D 融合得分
		if jazz_encoder is not None and prototype_jazz_1d is not None:
			emb_jazz = F.normalize(jazz_encoder(feat_full), dim=1)
			sim_jazz = torch.matmul(emb_jazz, prototype_jazz_1d.view(-1, 1)).squeeze()
			score[jazz_style_idx] = sim_jazz

	score_np = score.cpu().numpy().astype(float)
	indices = np.argsort(-score_np)
	ranked = [(class_names[i], float(score_np[i])) for i in indices]
	top1_style = ranked[0][0]
	confidence_report = _build_confidence_report(score)
	return top1_style, ranked, confidence_report


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="2D 模型预测脚本 (鼓点/音色双空间)")
	parser.add_argument("--audio_path", type=str, required=True, help="待预测音频文件路径")
	parser.add_argument("--data_root", type=str, default="", help="训练时使用的数据根目录, 为空则默认 Mymusic_all")
	parser.add_argument("--alpha_drum", type=float, default=None, help="可选: 手动指定全局鼓点权重, 覆盖模型中保存的自动权重")
	return parser.parse_args()


def main() -> None:
	# 无参数时进入交互模式
	if len(sys.argv) == 1:
		print("[2D-Predict] 进入交互模式, 将按提示设置参数 (直接回车使用默认值)")
		while True:
			audio_path = input("请输入待预测音频文件路径: ").strip()
			if audio_path:
				break
			print("路径不能为空, 请重新输入。")

		data_root_input = input(f"请输入数据根目录 (默认: {DEFAULT_MUSIC_ROOT}): ").strip()
		data_root = data_root_input or DEFAULT_MUSIC_ROOT

		print("\n[2D-Predict] 将使用以下设置进行预测:")
		print(f"  audio_path = {audio_path}")
		print(f"  data_root  = {data_root}")
		print("  alpha_drum = auto (默认使用每个风格自动学习到的鼓/音色权重)")

		print("[2D] 待预测音频:", audio_path)
		top1, ranked, confidence = predict_genre_2d(
			audio_path=audio_path,
			data_root=data_root,
			alpha_drum=None,
		)
		print("[2D] 预测结果:")
		print(f"  Top-1 风格: {top1}")
		print(
			f"  置信度: {confidence['confidence']:.3f} "
			f"(level={confidence['level']}, prob_margin={confidence['prob_margin']:.3f}, score_margin={confidence['score_margin']:.3f})"
		)
		if confidence["level"] == "low" and len(ranked) >= 2:
			print(f"  提示: 置信度较低, 更可能在 {ranked[0][0]} / {ranked[1][0]} 之间摇摆")
		print("\n[2D] 所有风格融合得分(从高到低):")
		for name, sc in ranked:
			print(f"  {name:20s}  score={sc:.4f}")
		return

	# 正常命令行参数模式
	args = parse_args()
	data_root = args.data_root or DEFAULT_MUSIC_ROOT
	print("[2D] 待预测音频:", args.audio_path)
	top1, ranked, confidence = predict_genre_2d(
		audio_path=args.audio_path,
		data_root=data_root,
		alpha_drum=args.alpha_drum,
	)
	print("[2D] 预测结果:")
	print(f"  Top-1 风格: {top1}")
	print(
		f"  置信度: {confidence['confidence']:.3f} "
		f"(level={confidence['level']}, prob_margin={confidence['prob_margin']:.3f}, score_margin={confidence['score_margin']:.3f})"
	)
	if confidence["level"] == "low" and len(ranked) >= 2:
		print(f"  提示: 置信度较低, 更可能在 {ranked[0][0]} / {ranked[1][0]} 之间摇摆")
	print("\n[2D] 所有风格融合得分(从高到低):")
	for name, sc in ranked:
		print(f"  {name:20s}  score={sc:.4f}")


if __name__ == "__main__":
	main()

