import argparse
import csv
import shutil
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


LINE_PATTERN = re.compile(r"^(?P<file>.+?)\tTop1:\s*(?P<top1>[^\t]+)\tTop\d+:\s*(?P<topk>[^\t]+)\tconf=(?P<conf>[0-9.]+)\tlevel=(?P<level>[^\t]+)")


@dataclass
class SongPred:
	file_name: str
	top1: str
	topk: List[str]
	confidence: float
	level: str


def _parse_pair_weights(raw: str) -> Dict[str, float]:
	"""Parse format like 'Plugg=2.0,New-Wave=1.8,Cloud=1.6'."""
	out: Dict[str, float] = {}
	if not raw.strip():
		return out
	for seg in raw.split(","):
		seg = seg.strip()
		if not seg or "=" not in seg:
			continue
		k, v = seg.split("=", 1)
		k = k.strip()
		try:
			out[k] = float(v.strip())
		except ValueError:
			continue
	return out


def _extract_target_folder(lines: List[str]) -> Optional[str]:
	for line in lines:
		if line.startswith("预测文件夹:"):
			path = line.split(":", 1)[1].strip()
			if path:
				return path
	return None


def _read_result_lines(result_txt: str) -> List[str]:
	if not os.path.isfile(result_txt):
		raise FileNotFoundError(f"目标txt文件不存在: {result_txt}")
	with open(result_txt, "r", encoding="utf-8") as f:
		return [x.rstrip("\n") for x in f]


def _infer_true_style(result_lines: List[str], style_override: str) -> str:
	if style_override.strip():
		return style_override.strip()
	folder = _extract_target_folder(result_lines)
	if not folder:
		raise ValueError("无法从结果文件中推断真实风格, 请通过 --true_style 指定。")
	return os.path.basename(os.path.normpath(folder))


def _parse_predictions(lines: List[str]) -> List[SongPred]:
	preds: List[SongPred] = []
	for line in lines:
		line = line.strip()
		if "\tTop1:" not in line:
			continue
		m = LINE_PATTERN.match(line)
		if not m:
			continue
		file_name = m.group("file").strip()
		top1 = m.group("top1").strip()
		topk = [x.strip() for x in m.group("topk").split(",") if x.strip()]
		confidence = float(m.group("conf"))
		level = m.group("level").strip()
		preds.append(SongPred(file_name=file_name, top1=top1, topk=topk, confidence=confidence, level=level))
	return preds


def _compute_weight(
	true_style: str,
	pred: SongPred,
	pair_weight_map: Dict[str, float],
	other_error_factor: float,
	max_weight: float,
) -> Tuple[float, str, bool]:
	"""Compute per-song sample weight for training.

	Rules:
	- Correct but uncertain samples still get slight upweight (hard positives).
	- Misclassified samples get larger weights.
	- Confusions to specific styles (e.g., Plugg/New-Wave) can be assigned higher factors.
	"""
	is_correct = pred.top1 == true_style
	conf = max(0.0, min(1.0, pred.confidence))

	if is_correct:
		# Hard positive: confidence lower => slightly higher weight, avoid over-amplifying easy positives.
		hardness = max(0.0, 0.70 - conf)
		weight = 1.0 + 2.0 * hardness
		reason = "correct-hard-positive" if hardness > 0 else "correct-easy"
	else:
		# Hard negative: increase by confusion target and by uncertainty.
		pair_factor = pair_weight_map.get(pred.top1, other_error_factor)
		hardness = 1.0 + max(0.0, 0.70 - conf) * 2.0
		weight = 1.8 * pair_factor * hardness
		reason = f"misclassified-to-{pred.top1}"

	weight = max(0.5, min(max_weight, weight))
	return weight, reason, is_correct


def build_weights(
	result_txt: str,
	true_style: str,
	pair_weight_map: Dict[str, float],
	other_error_factor: float,
	max_weight: float,
) -> Tuple[str, List[Dict[str, str]]]:
	lines = _read_result_lines(result_txt)

	inferred_true_style = _infer_true_style(lines, true_style)
	preds = _parse_predictions(lines)
	if not preds:
		raise ValueError("未解析到有效预测行, 请确认输入文件格式。")

	rows: List[Dict[str, str]] = []
	for p in preds:
		weight, reason, is_correct = _compute_weight(
			true_style=inferred_true_style,
			pred=p,
			pair_weight_map=pair_weight_map,
			other_error_factor=other_error_factor,
			max_weight=max_weight,
		)
		rows.append(
			{
				"file_name": p.file_name,
				"true_style": inferred_true_style,
				"pred_top1": p.top1,
				"pred_topk": "|".join(p.topk),
				"confidence": f"{p.confidence:.6f}",
				"level": p.level,
				"is_correct": "1" if is_correct else "0",
				"weight": f"{weight:.6f}",
				"reason": reason,
			}
		)

	return inferred_true_style, rows


def organize_misclassified_songs(
	result_txt: str,
	rows: List[Dict[str, str]],
	organize_root: str,
	coverage_threshold: float = 0.75,
	mode: str = "copy",
) -> Tuple[int, int, List[str], float]:
	"""Group misclassified songs into subfolders like TrueStyle_to_PredStyle.

	Selection strategy:
	1) Count correctly predicted songs as already-processed.
	2) Sort misclassified target styles by count, from high to low.
	3) Add one style at a time until processed/total >= coverage_threshold.

	Returns (moved_or_copied_count, missing_source_count, selected_styles, processed_ratio).
	"""
	lines = _read_result_lines(result_txt)
	target_folder = _extract_target_folder(lines)
	if not target_folder:
		raise ValueError("结果文件中未找到 预测文件夹 字段, 无法定位源歌曲目录。")
	if not os.path.isdir(target_folder):
		raise FileNotFoundError(f"源歌曲目录不存在: {target_folder}")
	if not organize_root.strip():
		# 默认直接放到真实风格目录下: TrueStyle/TrueStyle_to_PredStyle
		organize_root = target_folder

	mode = mode.strip().lower()
	if mode not in ("copy", "move"):
		raise ValueError("organize mode 仅支持 copy 或 move")
	coverage_threshold = max(0.0, min(1.0, float(coverage_threshold)))

	total = len(rows)
	if total <= 0:
		return 0, 0, [], 0.0

	true_style = rows[0]["true_style"]
	correct_count = sum(1 for r in rows if r["pred_top1"] == true_style)
	by_pred: Dict[str, int] = {}
	for r in rows:
		pred_top1 = r["pred_top1"]
		if pred_top1 == true_style:
			continue
		by_pred[pred_top1] = by_pred.get(pred_top1, 0) + 1

	selected_styles: List[str] = []
	processed = correct_count
	for pred_style, cnt in sorted(by_pred.items(), key=lambda x: x[1], reverse=True):
		if processed / total >= coverage_threshold:
			break
		selected_styles.append(pred_style)
		processed += cnt
	selected_style_set = set(selected_styles)
	processed_ratio = processed / total

	os.makedirs(organize_root, exist_ok=True)
	moved_or_copied = 0
	missing_source = 0

	for r in rows:
		true_style = r["true_style"]
		pred_top1 = r["pred_top1"]
		if pred_top1 == true_style:
			continue
		if pred_top1 not in selected_style_set:
			continue

		file_name = r["file_name"]
		src_path = os.path.join(target_folder, file_name)
		if not os.path.isfile(src_path):
			missing_source += 1
			continue

		subfolder = f"{true_style}_to_{pred_top1}"
		dst_dir = os.path.join(organize_root, subfolder)
		os.makedirs(dst_dir, exist_ok=True)
		dst_path = os.path.join(dst_dir, file_name)

		if mode == "copy":
			shutil.copy2(src_path, dst_path)
		else:
			shutil.move(src_path, dst_path)
		moved_or_copied += 1

	return moved_or_copied, missing_source, selected_styles, processed_ratio


def run_divide_pipeline(
	result_txt: str,
	true_style: str = "",
	pair_weights: str = "Plugg=2.0,New-Wave=1.8,Cloud=1.6,Regalia=1.6",
	other_error_factor: float = 1.3,
	max_weight: float = 6.0,
	output_csv: str = "",
	summary_txt: str = "",
	organize_errors: bool = False,
	organize_root: str = "",
	organize_threshold: float = 0.75,
	organize_mode: str = "copy",
) -> Tuple[str, str, str, int]:
	"""Run divide end-to-end and return (output_csv, summary_txt, true_style, sample_count)."""
	pair_weight_map = _parse_pair_weights(pair_weights)

	true_style_final, rows = build_weights(
		result_txt=result_txt,
		true_style=true_style,
		pair_weight_map=pair_weight_map,
		other_error_factor=other_error_factor,
		max_weight=max_weight,
	)

	base_dir = os.path.dirname(result_txt) or "."
	default_csv = os.path.join(base_dir, f"{true_style_final}_sample_weights.csv")
	default_summary = os.path.join(base_dir, f"{true_style_final}_weight_summary.txt")

	output_csv_final = output_csv or default_csv
	summary_txt_final = summary_txt or default_summary

	write_csv(output_csv_final, rows)
	write_summary(summary_txt_final, true_style_final, rows)

	if organize_errors:
		organize_root_final = organize_root
		if not organize_root_final.strip():
			lines = _read_result_lines(result_txt)
			target_folder = _extract_target_folder(lines)
			organize_root_final = target_folder or (os.path.dirname(result_txt) or ".")
		copied_count, missing_count, selected_styles, processed_ratio = organize_misclassified_songs(
			result_txt=result_txt,
			rows=rows,
			organize_root=organize_root_final,
			coverage_threshold=organize_threshold,
			mode=organize_mode,
		)
		print(f"[divide] 归类目录: {organize_root_final}")
		print(f"[divide] 归类阈值: {organize_threshold:.2f}")
		print(f"[divide] 已选错分风格: {selected_styles if selected_styles else '无'}")
		print(f"[divide] 已处理覆盖率: {processed_ratio:.2%}")
		print(f"[divide] 归类完成: {copied_count} 首")
		if missing_count > 0:
			print(f"[divide] 未找到源文件: {missing_count} 首")

	return output_csv_final, summary_txt_final, true_style_final, len(rows)


def write_csv(csv_path: str, rows: List[Dict[str, str]]) -> None:
	if not rows:
		return
	parent = os.path.dirname(csv_path)
	if parent:
		os.makedirs(parent, exist_ok=True)
	with open(csv_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
		writer.writeheader()
		writer.writerows(rows)


def write_summary(summary_path: str, true_style: str, rows: List[Dict[str, str]]) -> None:
	parent = os.path.dirname(summary_path)
	if parent:
		os.makedirs(parent, exist_ok=True)

	total = len(rows)
	correct = sum(1 for r in rows if r["is_correct"] == "1")
	incorrect = total - correct
	mean_weight = sum(float(r["weight"]) for r in rows) / max(total, 1)

	by_pred: Dict[str, int] = {}
	for r in rows:
		k = r["pred_top1"]
		by_pred[k] = by_pred.get(k, 0) + 1

	with open(summary_path, "w", encoding="utf-8") as f:
		f.write("==== hard-negative 权重统计 ====\n")
		f.write(f"真实风格: {true_style}\n")
		f.write(f"样本总数: {total}\n")
		f.write(f"Top1 正确: {correct}\n")
		f.write(f"Top1 错误: {incorrect}\n")
		f.write(f"平均权重: {mean_weight:.4f}\n")
		f.write("\n按预测 Top1 计数:\n")
		for k, v in sorted(by_pred.items(), key=lambda x: x[1], reverse=True):
			f.write(f"  {k:16s} {v}\n")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="根据文件夹预测结果生成每首歌训练权重 (hard negative friendly).")
	parser.add_argument("--result_txt", "--result-txt", dest="result_txt", type=str, default="", help="*_predict_result.txt 文件路径")
	parser.add_argument("--true_style", type=str, default="", help="真实风格名, 为空时自动从 预测文件夹 推断")
	parser.add_argument(
		"--pair_weights",
		"--pair-weights",
		dest="pair_weights",
		type=str,
		default="Plugg=2.0,New-Wave=1.8,Cloud=1.6,Regalia=1.6",
		help="重点混淆风格权重, 例如 'Plugg=2.0,New-Wave=1.8'",
	)
	parser.add_argument("--other_error_factor", "--other-error-factor", dest="other_error_factor", type=float, default=1.3, help="非重点混淆错分的默认系数")
	parser.add_argument("--max_weight", "--max-weight", dest="max_weight", type=float, default=6.0, help="权重上限")
	parser.add_argument("--output_csv", "--output-csv", dest="output_csv", type=str, default="", help="输出 csv 路径")
	parser.add_argument("--summary_txt", "--summary-txt", dest="summary_txt", type=str, default="", help="输出汇总 txt 路径")
	parser.add_argument("--organize_errors", "--organize-errors", dest="organize_errors", action="store_true", help="将错分歌曲按 true_to_pred 规则归类到子文件夹")
	parser.add_argument("--organize_root", "--organize-root", dest="organize_root", type=str, default="", help="归类输出根目录, 默认使用真实风格目录 (预测文件夹)")
	parser.add_argument("--organize_threshold", "--organize-threshold", dest="organize_threshold", type=float, default=0.75, help="错分风格归类覆盖率阈值 (0-1), 默认 0.75")
	parser.add_argument("--organize_mode", "--organize-mode", dest="organize_mode", type=str, default="copy", help="归类方式: copy 或 move, 默认 copy")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if not args.result_txt.strip():
		print("请输入预测结果txt文件路径 (例如 D:\\Music_Prediction\\Hyperpop_predict_result.txt):")
		args.result_txt = input("> ").strip().strip('"').strip("'")
	if not args.result_txt.strip():
		raise ValueError("未提供结果txt路径, 请使用 --result-txt 或在交互中输入。")
	output_csv, summary_txt, true_style, sample_count = run_divide_pipeline(
		result_txt=args.result_txt,
		true_style=args.true_style,
		pair_weights=args.pair_weights,
		other_error_factor=args.other_error_factor,
		max_weight=args.max_weight,
		output_csv=args.output_csv,
		summary_txt=args.summary_txt,
		organize_errors=bool(args.organize_errors),
		organize_root=args.organize_root,
		organize_threshold=args.organize_threshold,
		organize_mode=args.organize_mode,
	)

	print("[divide] 处理完成")
	print(f"[divide] 真实风格: {true_style}")
	print(f"[divide] 样本数: {sample_count}")
	print(f"[divide] 输出 CSV: {output_csv}")
	print(f"[divide] 输出汇总: {summary_txt}")


if __name__ == "__main__":
	main()
