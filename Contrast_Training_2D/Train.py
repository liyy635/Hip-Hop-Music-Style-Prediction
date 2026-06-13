import os
import sys
import math
import argparse
import random
from typing import List, Dict, Tuple

import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, Sampler


##############################
# 路径与全局配置
##############################


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_MUSIC_ROOT = os.path.join(BASE_DIR, "Mymusic_all")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "Contrast_Audio_Model_2D")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")
JAZZ_STYLE_NAME = "Jazz"


##############################
# 数据集: 同时输出“鼓点维度 + 音色维度”特征
##############################


class ContrastiveAudio2DDataset(Dataset):
	"""从根目录构建数据集, 每首歌随机截取一段后做 HPSS 分离, 分别提取

	- percussive(打击/鼓点) log-mel 频谱
	- harmonic(和声/音色) log-mel 频谱

	返回值: (feat_perc, feat_harm, label_idx)
	- feat_perc: torch.FloatTensor, (1, n_mels, T)
	- feat_harm: torch.FloatTensor, (1, n_mels, T)
	- label_idx: int
	"""

	def __init__(
		self,
		root_dir: str,
		segment_duration: float = 8.0,
		sr: int = 22050,
		n_mels: int = 128,
	) -> None:
		super().__init__()
		self.root_dir = root_dir
		self.segment_duration = float(segment_duration)
		self.sr = int(sr)
		self.n_mels = int(n_mels)

		if not os.path.isdir(self.root_dir):
			raise FileNotFoundError(f"找不到音频根目录: {self.root_dir}")

		self.classes: List[str] = []
		self.class_to_idx: Dict[str, int] = {}
		self.samples: List[Tuple[str, int]] = []

		self._scan_files()

	def _scan_files(self) -> None:
		styles = [
			name
			for name in os.listdir(self.root_dir)
			if os.path.isdir(os.path.join(self.root_dir, name))
		]
		styles = sorted(styles)
		if not styles:
			raise ValueError(f"在 {self.root_dir} 下未找到任何风格子文件夹")

		self.classes = styles
		self.class_to_idx = {c: i for i, c in enumerate(styles)}

		for style in styles:
			style_dir = os.path.join(self.root_dir, style)
			for root, _, files in os.walk(style_dir):
				for fn in files:
					if fn.lower().endswith(AUDIO_EXTS):
						path = os.path.join(root, fn)
						self.samples.append((path, self.class_to_idx[style]))

		if not self.samples:
			raise ValueError(f"在 {self.root_dir} 下未找到任何音频文件")

		print("[2D] 检测到的风格类别:")
		for c, idx in self.class_to_idx.items():
			print(f"  {idx}: {c}")
		print("[2D] 总音频样本数:", len(self.samples))

	def __len__(self) -> int:
		return len(self.samples)

	def _load_random_segment(self, path: str) -> np.ndarray:
		target_dur = self.segment_duration
		try:
			total_dur = librosa.get_duration(filename=path)
		except Exception:
			total_dur = None

		if total_dur is None or not math.isfinite(total_dur) or total_dur <= 0:
			y, _ = librosa.load(path, sr=self.sr, mono=True, duration=target_dur)
		else:
			if total_dur <= target_dur:
				offset = 0.0
				cur_dur = total_dur
			else:
				start_min = 0.1 * total_dur
				start_max_allowed = 0.9 * total_dur - target_dur
				if start_max_allowed <= start_min:
					start_min = 0.0
					start_max = max(0.0, total_dur - target_dur)
				else:
					start_max = start_max_allowed
				if start_max > start_min:
					offset = float(np.random.uniform(start_min, start_max))
				else:
					offset = float(start_min)
				cur_dur = target_dur
			y, _ = librosa.load(path, sr=self.sr, mono=True, offset=offset, duration=cur_dur)

		if y is None or len(y) == 0:
			T = int(self.segment_duration * self.sr)
			return np.zeros(T, dtype=np.float32)

		T = int(self.segment_duration * self.sr)
		y = y.astype(np.float32)
		max_val = np.max(np.abs(y)) if len(y) > 0 else 0.0
		if max_val > 0:
			y = y / max_val

		if len(y) < T:
			pad = np.zeros(T - len(y), dtype=np.float32)
			y = np.concatenate([y, pad], axis=0)
		elif len(y) > T:
			y = y[:T]
		return y

	def _waveform_to_2d_logmel(self, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
		"""对波形做 HPSS 分离, 分别计算 percussive/harmonic 的 log-mel 频谱。"""

		# HPSS 分离: 返回 harmonic, percussive
		harm, perc = librosa.effects.hpss(y)

		def _to_logmel(x: np.ndarray) -> torch.Tensor:
			S = librosa.feature.melspectrogram(
				y=x,
				sr=self.sr,
				n_mels=self.n_mels,
				fmin=20.0,
				fmax=self.sr / 2.0,
				n_fft=1024,
				hop_length=512,
				power=2.0,
			)
			S_db = librosa.power_to_db(S, ref=np.max)
			mu = float(np.mean(S_db))
			sigma = float(np.std(S_db)) + 1e-6
			S_norm = (S_db - mu) / sigma
			return torch.from_numpy(S_norm).unsqueeze(0).float()

		feat_harm = _to_logmel(harm)
		feat_perc = _to_logmel(perc)
		return feat_perc, feat_harm

	def _waveform_to_logmel(self, y: np.ndarray) -> torch.Tensor:
		"""对完整波形直接计算 log-mel 频谱, 用于 Jazz 的 1D 训练分支。"""

		S = librosa.feature.melspectrogram(
			y=y,
			sr=self.sr,
			n_mels=self.n_mels,
			fmin=20.0,
			fmax=self.sr / 2.0,
			n_fft=1024,
			hop_length=512,
			power=2.0,
		)
		S_db = librosa.power_to_db(S, ref=np.max)
		mu = float(np.mean(S_db))
		sigma = float(np.std(S_db)) + 1e-6
		S_norm = (S_db - mu) / sigma
		return torch.from_numpy(S_norm).unsqueeze(0).float()

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
		path, label = self.samples[idx]
		y = self._load_random_segment(path)
		feat_perc, feat_harm = self._waveform_to_2d_logmel(y)
		feat_full = self._waveform_to_logmel(y)
		return feat_perc, feat_harm, feat_full, int(label)


class ClassBalancedBatchSampler(Sampler[List[int]]):
	"""按类别均衡采样的 batch sampler。

	每个 batch 由 `classes_per_batch` 个类别组成, 每个类别抽取
	`samples_per_class` 个样本 (样本不足时允许有放回采样)。
	"""

	def __init__(
		self,
		labels: List[int],
		classes_per_batch: int,
		samples_per_class: int,
		batches_per_epoch: int,
	) -> None:
		super().__init__()
		if not labels:
			raise ValueError("labels 不能为空")
		if classes_per_batch <= 0:
			raise ValueError("classes_per_batch 必须为正整数")
		if samples_per_class <= 0:
			raise ValueError("samples_per_class 必须为正整数")
		if batches_per_epoch <= 0:
			raise ValueError("batches_per_epoch 必须为正整数")

		self.labels = [int(x) for x in labels]
		self.classes_per_batch = int(classes_per_batch)
		self.samples_per_class = int(samples_per_class)
		self.batches_per_epoch = int(batches_per_epoch)

		indices_per_class: Dict[int, List[int]] = {}
		for idx, c in enumerate(self.labels):
			indices_per_class.setdefault(int(c), []).append(idx)

		self.indices_per_class = indices_per_class
		self.class_ids = sorted(self.indices_per_class.keys())
		if not self.class_ids:
			raise ValueError("未找到可用于采样的类别")

	def __len__(self) -> int:
		return self.batches_per_epoch

	def __iter__(self):
		num_classes = len(self.class_ids)
		classes_per_batch = min(self.classes_per_batch, num_classes)

		for _ in range(self.batches_per_epoch):
			selected_classes = random.sample(self.class_ids, classes_per_batch)
			batch_indices: List[int] = []

			for c in selected_classes:
				pool = self.indices_per_class[c]
				if len(pool) >= self.samples_per_class:
					picked = random.sample(pool, self.samples_per_class)
				else:
					picked = [random.choice(pool) for _ in range(self.samples_per_class)]
				batch_indices.extend(picked)

			random.shuffle(batch_indices)
			yield batch_indices


##############################
# 模型: 共用结构, 两个编码器分鼓点 / 音色
##############################


class AudioEncoder(nn.Module):
	def __init__(self, n_mels: int = 128, embed_dim: int = 256) -> None:
		super().__init__()

		def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
			return nn.Sequential(
				nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(out_ch),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
				nn.BatchNorm2d(out_ch),
				nn.ReLU(inplace=True),
				nn.MaxPool2d(kernel_size=(2, 2)),
			)

		self.n_mels = n_mels
		self.embed_dim = embed_dim

		self.block1 = conv_block(1, 64)
		self.block2 = conv_block(64, 128)
		self.block3 = conv_block(128, 256)
		self.block4 = conv_block(256, 512)

		self.fc = nn.Sequential(
			nn.Linear(512 * 2, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, embed_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: (B, 1, n_mels, T)
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)
		# (B, C, F, T)
		x_mean = torch.mean(x, dim=[2, 3])
		x_max, _ = torch.max(x, dim=2)
		x_max, _ = torch.max(x_max, dim=2)
		x = torch.cat([x_mean, x_max], dim=1)
		emb = self.fc(x)
		return emb


##############################
# 监督对比损失 (与 1D 版本相同)
##############################


def supervised_contrastive_loss(
	features: torch.Tensor,
	labels: torch.Tensor,
	temperature: float = 0.1,
	class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
	device = features.device
	batch_size = features.shape[0]
	if batch_size < 2:
		return torch.tensor(0.0, device=device, requires_grad=True)

	feats = F.normalize(features, dim=1)
	similarity_matrix = torch.div(torch.matmul(feats, feats.T), temperature)

	logits_max, _ = similarity_matrix.max(dim=1, keepdim=True)
	logits = similarity_matrix - logits_max.detach()

	labels = labels.contiguous().view(-1, 1)
	mask = torch.eq(labels, labels.T).float().to(device)

	logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
	mask = mask * logits_mask

	exp_logits = torch.exp(logits) * logits_mask
	log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

	mask_sum = mask.sum(dim=1)
	mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask_sum + 1e-12)
	valid_mask = mask_sum > 0
	if not valid_mask.any():
		return torch.tensor(0.0, device=device)

	loss_vec = -mean_log_prob_pos[valid_mask]
	if class_weights is None:
		return loss_vec.mean()

	labels_flat = labels.view(-1)
	sample_weights = class_weights.to(device)[labels_flat][valid_mask]
	weight_sum = sample_weights.sum()
	if weight_sum.item() <= 0:
		return loss_vec.mean()
	return (loss_vec * sample_weights).sum() / weight_sum


##############################
# 训练: 鼓点 / 音色 双 encoder + 双原型
##############################


def _get_checkpoint_path(data_root: str) -> str:
	folder_name = os.path.basename(os.path.normpath(data_root)) or "dataset"
	file_name = f"contrast_audio_encoder_2d_{folder_name}.pt"
	return os.path.join(CHECKPOINT_DIR, file_name)


def train_contrastive_2d(
	num_epochs: int = 30,
	batch_size: int = 32,
	learning_rate: float = 1e-4,
	segment_duration: float = 8.0,
	sr: int = 22050,
	n_mels: int = 128,
	embed_dim: int = 256,
	temperature: float = 0.1,
	alpha_drum: float = 0.5,
	train_ratio: float = 0.8,
	balanced_batch: bool = True,
	classes_per_batch: int = 8,
	samples_per_class: int = 6,
	data_root: str = DEFAULT_MUSIC_ROOT,
) -> Dict:
	"""2D 对比学习训练: 鼓点 / 音色两个 encoder, 各自对比学习 + 各自原型。"""

	random.seed(42)
	np.random.seed(42)
	torch.manual_seed(42)

	if not os.path.isdir(data_root):
		raise FileNotFoundError(f"音频根目录不存在: {data_root}")

	print("[2D] 使用的音频根目录:", data_root)

	dataset = ContrastiveAudio2DDataset(
		root_dir=data_root,
		segment_duration=segment_duration,
		sr=sr,
		n_mels=n_mels,
	)

	class_names = dataset.classes
	num_classes = len(class_names)
	print("[2D] 风格类别数:", num_classes)

	# 按风格划分训练 / 测试索引
	train_ratio = float(train_ratio)
	if train_ratio < 0.0:
		train_ratio = 0.0
	if train_ratio > 1.0:
		train_ratio = 1.0

	indices_per_class: Dict[int, List[int]] = {i: [] for i in range(num_classes)}
	for idx, (_, label_idx) in enumerate(dataset.samples):
		indices_per_class[int(label_idx)].append(idx)

	train_indices: List[int] = []
	test_indices: List[int] = []
	for c in range(num_classes):
		idxs = indices_per_class[c]
		if not idxs:
			continue
		random.shuffle(idxs)
		n_total = len(idxs)
		if train_ratio >= 1.0 or n_total <= 1:
			# 全部用于训练, 或只有一个样本无法再划分
			train_indices.extend(idxs)
		else:
			n_train = int(n_total * train_ratio)
			# 至少保证每类有 1 个训练样本和 1 个测试样本
			n_train = max(1, min(n_total - 1, n_train))
			train_indices.extend(idxs[:n_train])
			test_indices.extend(idxs[n_train:])

	train_indices = sorted(train_indices)
	test_indices = sorted(test_indices)
	print(f"[2D] 按 train_ratio={train_ratio:.2f} 划分数据: 训练样本 {len(train_indices)} 条, 测试样本 {len(test_indices)} 条")
	if not test_indices:
		print("[2D] 警告: 当前划分没有产生测试集 (train_ratio>=1 或样本过少), 后续测试脚本将无法使用本次划分。")

	train_dataset = Subset(dataset, train_indices) if train_indices else dataset
	test_samples = [(dataset.samples[i][0], int(dataset.samples[i][1])) for i in test_indices]

	base_weight = 1.0
	extra_weight = 2.0
	class_weights_list: List[float] = []
	for name in class_names:
		if name in {"New-Wave", "Gospel", "Emo", "Mumble"}:
			class_weights_list.append(extra_weight)
		else:
			class_weights_list.append(base_weight)
	print("[2D] 类别权重设置:")
	for idx, (name, w) in enumerate(zip(class_names, class_weights_list)):
		print(f"  [{idx}] {name}: weight={w}")

	train_labels = [int(dataset.samples[i][1]) for i in train_indices]
	if balanced_batch:
		if classes_per_batch <= 0:
			classes_per_batch = num_classes
		classes_per_batch = min(int(classes_per_batch), num_classes)
		samples_per_class = max(1, int(samples_per_class))
		effective_batch_size = classes_per_batch * samples_per_class
		batches_per_epoch = max(1, len(train_indices) // effective_batch_size)

		batch_sampler = ClassBalancedBatchSampler(
			labels=train_labels,
			classes_per_batch=classes_per_batch,
			samples_per_class=samples_per_class,
			batches_per_epoch=batches_per_epoch,
		)
		dataloader = DataLoader(
			train_dataset,
			batch_sampler=batch_sampler,
			num_workers=4,
			pin_memory=True,
		)
		print(
			f"[2D] 使用类均衡采样: 每批 {classes_per_batch} 类 x 每类 {samples_per_class} 首, "
			f"有效 batch_size={effective_batch_size}, 每轮 batch 数={batches_per_epoch}"
		)
	else:
		dataloader = DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=4,
			pin_memory=True,
		)
		print(f"[2D] 使用普通随机采样: batch_size={batch_size}")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("[2D] 使用设备:", device)
	class_weights = torch.tensor(class_weights_list, dtype=torch.float32, device=device)

	drum_encoder = AudioEncoder(n_mels=n_mels, embed_dim=embed_dim).to(device)
	timbre_encoder = AudioEncoder(n_mels=n_mels, embed_dim=embed_dim).to(device)
	jazz_encoder = AudioEncoder(n_mels=n_mels, embed_dim=embed_dim).to(device)
	optimizer = torch.optim.Adam(
		list(drum_encoder.parameters())
		+ list(timbre_encoder.parameters())
		+ list(jazz_encoder.parameters()),
		lr=learning_rate,
	)
	jazz_style_idx = class_names.index(JAZZ_STYLE_NAME) if JAZZ_STYLE_NAME in class_names else None
	if jazz_style_idx is None:
		print(f"[2D] 提示: 当前数据集中未找到风格 {JAZZ_STYLE_NAME}, 将仅使用原2D训练逻辑。")
	else:
		print(f"[2D] 已启用 {JAZZ_STYLE_NAME} 专用1D训练分支, 类别索引: {jazz_style_idx}")

	drum_encoder.train()
	timbre_encoder.train()
	jazz_encoder.train()
	for epoch in range(1, num_epochs + 1):
		total_loss = 0.0
		num_batches = 0

		for feat_perc, feat_harm, feat_full, labels in dataloader:
			feat_perc = feat_perc.to(device)
			feat_harm = feat_harm.to(device)
			feat_full = feat_full.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			emb_drum = drum_encoder(feat_perc)
			emb_timbre = timbre_encoder(feat_harm)
			emb_jazz = jazz_encoder(feat_full)

			loss_drum = supervised_contrastive_loss(
				features=emb_drum,
				labels=labels,
				temperature=temperature,
				class_weights=class_weights,
			)
			loss_timbre = supervised_contrastive_loss(
				features=emb_timbre,
				labels=labels,
				temperature=temperature,
				class_weights=class_weights,
			)
			loss = loss_drum + loss_timbre

			# 仅对 Jazz 样本施加 1D 对比训练, 其余风格保持原 2D 训练方式不变
			if jazz_style_idx is not None:
				jazz_mask = labels == int(jazz_style_idx)
				if torch.sum(jazz_mask) >= 2:
					jazz_labels = torch.zeros_like(labels[jazz_mask])
					loss_jazz = supervised_contrastive_loss(
						features=emb_jazz[jazz_mask],
						labels=jazz_labels,
						temperature=temperature,
						class_weights=None,
					)
					loss = loss + loss_jazz

			loss.backward()
			optimizer.step()

			total_loss += float(loss.item())
			num_batches += 1

		avg_loss = total_loss / max(1, num_batches)
		print(f"[2D] Epoch {epoch:03d}/{num_epochs:03d}  Loss(drum+timbre): {avg_loss:.4f}")

	# 计算鼓点 / 音色两个子空间各自的类别原型
	drum_encoder.eval()
	timbre_encoder.eval()
	jazz_encoder.eval()
	with torch.no_grad():
		sums_drum = [torch.zeros(embed_dim, device=device) for _ in range(num_classes)]
		sums_timbre = [torch.zeros(embed_dim, device=device) for _ in range(num_classes)]
		counts = [0 for _ in range(num_classes)]
		sum_jazz = torch.zeros(embed_dim, device=device)
		count_jazz = 0

		proto_loader = DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=4,
			pin_memory=True,
		)

		for feat_perc, feat_harm, feat_full, labels in proto_loader:
			feat_perc = feat_perc.to(device)
			feat_harm = feat_harm.to(device)
			feat_full = feat_full.to(device)
			labels = labels.to(device)

			emb_d = F.normalize(drum_encoder(feat_perc), dim=1)
			emb_t = F.normalize(timbre_encoder(feat_harm), dim=1)
			emb_j = F.normalize(jazz_encoder(feat_full), dim=1)

			for i in range(emb_d.shape[0]):
				c = int(labels[i].item())
				sums_drum[c] += emb_d[i]
				sums_timbre[c] += emb_t[i]
				counts[c] += 1
				if jazz_style_idx is not None and c == int(jazz_style_idx):
					sum_jazz += emb_j[i]
					count_jazz += 1

		prototypes_drum = []
		prototypes_timbre = []
		for c in range(num_classes):
			if counts[c] == 0:
				print(f"[2D] 警告: 类别 {class_names[c]} 没有样本, 原型向量置为 0")
				pd = torch.zeros(embed_dim, device=device)
				pt = torch.zeros(embed_dim, device=device)
			else:
				pd = sums_drum[c] / counts[c]
				pt = sums_timbre[c] / counts[c]
			pd = F.normalize(pd, dim=0)
			pt = F.normalize(pt, dim=0)
			prototypes_drum.append(pd)
			prototypes_timbre.append(pt)

		prototypes_drum = torch.stack(prototypes_drum, dim=0)
		prototypes_timbre = torch.stack(prototypes_timbre, dim=0)

		if jazz_style_idx is not None and count_jazz > 0:
			prototype_jazz_1d = F.normalize(sum_jazz / float(count_jazz), dim=0)
		else:
			prototype_jazz_1d = None

		# 基于训练好的 2D 空间, 量化每个风格在鼓/音色子空间的“对比分辨力”
		# 使用: 类内平均距离(样本到本类原型) 与 类间平均距离(样本到其它类原型)
		intra_drum = torch.zeros(num_classes, device=device)
		intra_timbre = torch.zeros(num_classes, device=device)
		inter_drum = torch.zeros(num_classes, device=device)
		inter_timbre = torch.zeros(num_classes, device=device)
		count_per_class = torch.zeros(num_classes, device=device)

		for feat_perc, feat_harm, _, labels in proto_loader:
			feat_perc = feat_perc.to(device)
			feat_harm = feat_harm.to(device)
			labels = labels.to(device)

			emb_d = F.normalize(drum_encoder(feat_perc), dim=1)
			emb_t = F.normalize(timbre_encoder(feat_harm), dim=1)

			# 与所有类别原型的相似度 (B, C)
			sims_d = torch.matmul(emb_d, prototypes_drum.T)
			sims_t = torch.matmul(emb_t, prototypes_timbre.T)

			batch_size_cur = labels.shape[0]
			idx_range = torch.arange(batch_size_cur, device=device)

			# 类内距离: 样本到自身类别原型的 1 - cos
			intra_d = 1.0 - sims_d[idx_range, labels]
			intra_t = 1.0 - sims_t[idx_range, labels]

			# 类间距离: 样本到其它类别原型的平均 1 - cos
			mask = torch.ones_like(sims_d, dtype=torch.bool)
			mask[idx_range, labels] = False
			dist_d = (1.0 - sims_d)[mask].view(batch_size_cur, -1)
			dist_t = (1.0 - sims_t)[mask].view(batch_size_cur, -1)
			inter_d = dist_d.mean(dim=1)
			inter_t = dist_t.mean(dim=1)

			# 按类别累计
			intra_drum.index_add_(0, labels, intra_d)
			intra_timbre.index_add_(0, labels, intra_t)
			inter_drum.index_add_(0, labels, inter_d)
			inter_timbre.index_add_(0, labels, inter_t)
			count_per_class.index_add_(0, labels, torch.ones_like(intra_d))

		# 避免除零
		count_safe = count_per_class.clamp_min(1e-6)
		intra_drum = intra_drum / count_safe
		intra_timbre = intra_timbre / count_safe
		inter_drum = inter_drum / count_safe
		inter_timbre = inter_timbre / count_safe

		# 对比分辨力: q = inter - intra, 越大表示该子空间越适合做该类的对比指标
		q_drum = inter_drum - intra_drum
		q_timbre = inter_timbre - intra_timbre

		# 只使用非负部分, 然后归一化得到每个风格的鼓/音色权重
		q_drum_clamped = torch.clamp(q_drum, min=0.0)
		q_timbre_clamped = torch.clamp(q_timbre, min=0.0)
		den = q_drum_clamped + q_timbre_clamped + 1e-6
		alpha_drum_per_class = q_drum_clamped / den
		alpha_timbre_per_class = 1.0 - alpha_drum_per_class

		print("[2D] 按风格分析得到的鼓/音色权重 (alpha_drum):")
		for idx, name in enumerate(class_names):
			print(
				f"  [{idx}] {name:20s}  q_drum={q_drum[idx].item():.4f}  "
				f"q_timbre={q_timbre[idx].item():.4f}  alpha_drum={alpha_drum_per_class[idx].item():.3f}"
			)

	prototypes_drum_cpu = prototypes_drum.cpu()
	prototypes_timbre_cpu = prototypes_timbre.cpu()
	alpha_drum_per_class_cpu = alpha_drum_per_class.cpu()
	alpha_timbre_per_class_cpu = alpha_timbre_per_class.cpu()

	state = {
		"drum_encoder_state_dict": drum_encoder.state_dict(),
		"timbre_encoder_state_dict": timbre_encoder.state_dict(),
		"jazz_encoder_state_dict": jazz_encoder.state_dict(),
		"class_names": class_names,
		"prototypes_drum": prototypes_drum_cpu,
		"prototypes_timbre": prototypes_timbre_cpu,
		"jazz_style_name": JAZZ_STYLE_NAME,
		"jazz_style_idx": int(jazz_style_idx) if jazz_style_idx is not None else -1,
		"prototype_jazz_1d": prototype_jazz_1d.cpu() if prototype_jazz_1d is not None else None,
		"alpha_drum_per_class": alpha_drum_per_class_cpu,
		"alpha_timbre_per_class": alpha_timbre_per_class_cpu,
		"q_drum": q_drum.cpu(),
		"q_timbre": q_timbre.cpu(),
		"test_samples": test_samples,
		"sr": sr,
		"n_mels": n_mels,
		"segment_duration": segment_duration,
		"embed_dim": embed_dim,
		"alpha_drum": float(alpha_drum),
	}

	ckpt_path = _get_checkpoint_path(data_root)
	torch.save(state, ckpt_path)
	print("[2D] 训练完成, 已保存 2D 编码器与原型到:", ckpt_path)
	return state


##############################
# 命令行入口 (仅训练)
##############################


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="2D 对比学习: 仅训练脚本 (含按比例划分训练/测试集)")
	parser.add_argument("--epochs", type=int, default=30)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--segment_duration", type=float, default=8.0)
	parser.add_argument("--sr", type=int, default=22050)
	parser.add_argument("--n_mels", type=int, default=128)
	parser.add_argument("--embed_dim", type=int, default=256)
	parser.add_argument("--temperature", type=float, default=0.1)
	parser.add_argument("--alpha_drum", type=float, default=0.5, help="鼓点维度得分权重, 仅用于保存到 checkpoint 作为默认值")
	parser.add_argument("--train_ratio", type=float, default=0.8, help="每个风格用于训练的比例, 剩余 1-train_ratio 作为测试集")
	parser.add_argument("--balanced_batch", action="store_true", default=True, help="启用类均衡采样: 每个 batch 固定若干类别且每类固定样本数 (默认启用)")
	parser.add_argument("--disable_balanced_batch", action="store_true", help="关闭类均衡采样, 回退到普通随机 batch")
	parser.add_argument("--classes_per_batch", type=int, default=8, help="类均衡采样时, 每个 batch 抽取的类别数")
	parser.add_argument("--samples_per_class", type=int, default=6, help="类均衡采样时, 每个类别在 batch 中抽取的样本数")
	parser.add_argument("--data_root", type=str, default="", help="数据根目录, 为空则使用默认 Mymusic_all")
	return parser.parse_args()


def main() -> None:
	# 如果没有提供命令行参数, 进入交互模式, 主动询问关键参数
	if len(sys.argv) == 1:
		print("[2D-Train] 进入交互模式, 将按提示设置参数 (直接回车使用默认值)")
		data_root_input = input(f"请输入数据根目录 (默认: {DEFAULT_MUSIC_ROOT}): ").strip()
		data_root = data_root_input or DEFAULT_MUSIC_ROOT

		train_ratio_default = 0.8
		train_ratio_str = input(f"每个风格用于训练的比例 train_ratio (0-1, 默认 {train_ratio_default}): ").strip()
		try:
			train_ratio = float(train_ratio_str) if train_ratio_str else train_ratio_default
		except ValueError:
			print("输入无效, 使用默认 train_ratio=0.8")
			train_ratio = train_ratio_default

		print("\n[2D-Train] 将使用以下设置进行训练:")
		print(f"  data_root   = {data_root}")
		print(f"  train_ratio = {train_ratio:.2f} (训练集比例)")
		print("  balanced_batch = True")
		print("  classes_per_batch = 8")
		print("  samples_per_class = 6")
		print(f"  alpha_drum  = 0.50 (鼓点子空间全局默认值, 实际预测时优先使用每风格自动学习的权重)")

		train_contrastive_2d(
			num_epochs=30,
			batch_size=32,
			learning_rate=1e-4,
			segment_duration=8.0,
			sr=22050,
			n_mels=128,
			embed_dim=256,
			temperature=0.1,
			alpha_drum=0.5,
			train_ratio=train_ratio,
			balanced_batch=True,
			classes_per_batch=8,
			samples_per_class=6,
			data_root=data_root,
		)
		return

	# 正常命令行参数模式
	args = parse_args()
	data_root = args.data_root or DEFAULT_MUSIC_ROOT
	train_contrastive_2d(
		num_epochs=args.epochs,
		batch_size=args.batch_size,
		learning_rate=args.lr,
		segment_duration=args.segment_duration,
		sr=args.sr,
		n_mels=args.n_mels,
		embed_dim=args.embed_dim,
		temperature=args.temperature,
		alpha_drum=args.alpha_drum,
		train_ratio=args.train_ratio,
		balanced_batch=not bool(args.disable_balanced_batch),
		classes_per_batch=args.classes_per_batch,
		samples_per_class=args.samples_per_class,
		data_root=data_root,
	)


if __name__ == "__main__":
	main()

