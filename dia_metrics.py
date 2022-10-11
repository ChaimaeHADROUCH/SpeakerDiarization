# This code comes from pyannote.audio
# All credits go to the pyannote.audio author(s)
#
# The code is not imported because of errors from
# a part of pyannote.audio that we don't use here.

from numbers import Number
from typing import Optional, Tuple, Union, Callable, List

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric


def mse_cost_func(Y, y, **kwargs):
    """Compute class-wise mean-squared error

    Parameters
    ----------
    Y, y : (num_frames, num_classes) torch.tensor

    Returns
    -------
    mse : (num_classes, ) torch.tensor
        Mean-squared error
    """
    return torch.mean(F.mse_loss(Y, y, reduction="none"), axis=0)


def permutate(
    y1: torch.Tensor,
    y2: torch.Tensor,
    cost_func: Optional[Callable] = None,
    return_cost: bool = False,
) -> Tuple[torch.Tensor, List[Tuple[int]]]:

    batch_size, num_samples, num_classes_1 = y1.shape

    if len(y2.shape) == 2:
        y2 = y2.expand(batch_size, -1, -1)

    if len(y2.shape) != 3:
        msg = "Incorrect shape: should be (batch_size, num_frames, num_classes)."
        raise ValueError(msg)

    batch_size_, num_samples_, num_classes_2 = y2.shape
    if batch_size != batch_size_ or num_samples != num_samples_:
        msg = f"Shape mismatch: {tuple(y1.shape)} vs. {tuple(y2.shape)}."
        raise ValueError(msg)

    if cost_func is None:
        cost_func = mse_cost_func

    permutations = []
    permutated_y2 = []

    if return_cost:
        costs = []

    permutated_y2 = torch.zeros(y1.shape, device=y2.device, dtype=y2.dtype)

    for b, (y1_, y2_) in enumerate(zip(y1, y2)):
        # y1_ is (num_samples, num_classes_1)-shaped
        # y2_ is (num_samples, num_classes_2)-shaped
        with torch.no_grad():
            cost = torch.stack(
                [
                    cost_func(y2_, y1_[:, i : i + 1].expand(-1, num_classes_2))
                    for i in range(num_classes_1)
                ],
            )

        if num_classes_2 > num_classes_1:
            padded_cost = F.pad(
                cost,
                (0, 0, 0, num_classes_2 - num_classes_1),
                "constant",
                torch.max(cost) + 1,
            )
        else:
            padded_cost = cost

        permutation = [None] * num_classes_1
        for k1, k2 in zip(*linear_sum_assignment(padded_cost.cpu())):
            if k1 < num_classes_1:
                permutation[k1] = k2
                permutated_y2[b, :, k1] = y2_[:, k2]
        permutations.append(tuple(permutation))

        if return_cost:
            costs.append(cost)

    if return_cost:
        return permutated_y2, permutations, torch.stack(costs)

    return permutated_y2, permutations


def _der_update(
    preds: torch.Tensor,
    target: torch.Tensor,
    threshold: Union[torch.Tensor, float] = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute components of diarization error rate

    Parameters
    ----------
    preds : torch.Tensor
        (batch_size, num_speakers, num_frames)-shaped continuous predictions.
    target : torch.Tensor
        (batch_size, num_speakers, num_frames)-shaped (0 or 1) targets.
    threshold : float or torch.Tensor, optional
        Threshold(s) used to binarize predictions. Defaults to 0.5.

    Returns
    -------
    false_alarm : (num_thresholds, )-shaped torch.Tensor
    missed_detection : (num_thresholds, )-shaped torch.Tensor
    speaker_confusion : (num_thresholds, )-shaped torch.Tensor
    speech_total : torch.Tensor
        Diarization error rate components accumulated over the whole batch.
    """

    # make threshold a (num_thresholds,) tensor
    scalar_threshold = isinstance(threshold, Number)
    if scalar_threshold:
        threshold = torch.tensor([threshold], dtype=preds.dtype, device=preds.device)

    # find the optimal mapping between target and (soft) predictions
    permutated_preds, _ = permutate(
        torch.transpose(target, 1, 2), torch.transpose(preds, 1, 2)
    )
    permutated_preds = torch.transpose(permutated_preds, 1, 2)
    # (batch_size, num_speakers, num_frames)

    # turn continuous [0, 1] predictions into binary {0, 1} decisions
    hypothesis = (permutated_preds.unsqueeze(-1) > threshold).float()
    # (batch_size, num_speakers, num_frames, num_thresholds)

    target = target.unsqueeze(-1)
    # (batch_size, num_speakers, num_frames, 1)

    detection_error = torch.sum(hypothesis, 1) - torch.sum(target, 1)
    # (batch_size, num_frames, num_thresholds)

    false_alarm = torch.maximum(detection_error, torch.zeros_like(detection_error))
    # (batch_size, num_frames, num_thresholds)

    missed_detection = torch.maximum(
        -detection_error, torch.zeros_like(detection_error)
    )
    # (batch_size, num_frames, num_thresholds)

    speaker_confusion = torch.sum((hypothesis != target) * hypothesis, 1) - false_alarm
    # (batch_size, num_frames, num_thresholds)

    false_alarm = torch.sum(torch.sum(false_alarm, 1), 0)
    missed_detection = torch.sum(torch.sum(missed_detection, 1), 0)
    speaker_confusion = torch.sum(torch.sum(speaker_confusion, 1), 0)
    # (num_thresholds, )

    speech_total = 1.0 * torch.sum(target)

    if scalar_threshold:
        false_alarm = false_alarm[0]
        missed_detection = missed_detection[0]
        speaker_confusion = speaker_confusion[0]

    return false_alarm, missed_detection, speaker_confusion, speech_total


def _der_compute(
    false_alarm: torch.Tensor,
    missed_detection: torch.Tensor,
    speaker_confusion: torch.Tensor,
    speech_total: torch.Tensor,
) -> torch.Tensor:
    """Compute diarization error rate from its components

    Parameters
    ----------
    false_alarm : (num_thresholds, )-shaped torch.Tensor
    missed_detection : (num_thresholds, )-shaped torch.Tensor
    speaker_confusion : (num_thresholds, )-shaped torch.Tensor
    speech_total : torch.Tensor
        Diarization error rate components, in number of frames.

    Returns
    -------
    der : (num_thresholds, )-shaped torch.Tensor
        Diarization error rate.
    """

    # TODO: handle corner case where speech_total == 0
    return (false_alarm + missed_detection + speaker_confusion) / speech_total


class DiarizationErrorRate(Metric):
    """Diarization error rate

    Parameters
    ----------
    threshold : float, optional
        Threshold used to binarize predictions. Defaults to 0.5.

    Notes
    -----
    While pyannote.audio conventions is to store speaker activations with
    (batch_size, num_frames, num_speakers)-shaped tensors, this torchmetrics metric
    expects them to be shaped as (batch_size, num_speakers, num_frames) tensors.
    """

    higher_is_better = False
    is_differentiable = False

    def __init__(self, threshold: float = 0.5):
        super().__init__()

        self.threshold = threshold

        self.add_state("false_alarm", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "missed_detection", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "speaker_confusion", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("speech_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Compute and accumulate components of diarization error rate

        Parameters
        ----------
        preds : torch.Tensor
            (batch_size, num_speakers, num_frames)-shaped continuous predictions.
        target : torch.Tensor
            (batch_size, num_speakers, num_frames)-shaped (0 or 1) targets.

        Returns
        -------
        false_alarm : torch.Tensor
        missed_detection : torch.Tensor
        speaker_confusion : torch.Tensor
        speech_total : torch.Tensor
            Diarization error rate components accumulated over the whole batch.
        """

        false_alarm, missed_detection, speaker_confusion, speech_total = _der_update(
            preds, target, threshold=self.threshold
        )
        self.false_alarm += false_alarm
        self.missed_detection += missed_detection
        self.speaker_confusion += speaker_confusion
        self.speech_total += speech_total

    def compute(self):
        return _der_compute(
            self.false_alarm,
            self.missed_detection,
            self.speaker_confusion,
            self.speech_total,
        )


class SpeakerConfusionRate(DiarizationErrorRate):
    def compute(self):
        # TODO: handler corner case where speech_total == 0
        return self.speaker_confusion / self.speech_total


class FalseAlarmRate(DiarizationErrorRate):
    def compute(self):
        # TODO: handler corner case where speech_total == 0
        return self.false_alarm / self.speech_total


class MissedDetectionRate(DiarizationErrorRate):
    def compute(self):
        # TODO: handler corner case where speech_total == 0
        return self.missed_detection / self.speech_total


class OptimalDiarizationErrorRate(Metric):
    """Optiml Diarization error rate

    Parameters
    ----------
    thresholds : torch.Tensor, optional
        Thresholds used to binarize predictions.
        Defaults to torch.linspace(0.0, 1.0, 51)

    Notes
    -----
    While pyannote.audio conventions is to store speaker activations with
    (batch_size, num_frames, num_speakers)-shaped tensors, this torchmetrics metric
    expects them to be shaped as (batch_size, num_speakers, num_frames) tensors.
    """

    higher_is_better = False
    is_differentiable = False

    def __init__(self, threshold: Optional[torch.Tensor] = None):
        super().__init__()

        threshold = threshold or torch.linspace(0.0, 1.0, 51)
        self.add_state("threshold", default=threshold, dist_reduce_fx="mean")
        (num_thresholds,) = threshold.shape

        # note that CamelCase is used to indicate that those states contain values for multiple thresholds
        # this is for torchmetrics to know that these states are different from those of DiarizationErrorRate
        # for which only one threshold is used.

        self.add_state(
            "FalseAlarm",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "MissedDetection",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "SpeakerConfusion",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )
        self.add_state("speech_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Compute and accumulate components of diarization error rate

        Parameters
        ----------
        preds : torch.Tensor
            (batch_size, num_speakers, num_frames)-shaped continuous predictions.
        target : torch.Tensor
            (batch_size, num_speakers, num_frames)-shaped (0 or 1) targets.

        Returns
        -------
        false_alarm : torch.Tensor
        missed_detection : torch.Tensor
        speaker_confusion : torch.Tensor
        speech_total : torch.Tensor
            Diarization error rate components accumulated over the whole batch.
        """

        false_alarm, missed_detection, speaker_confusion, speech_total = _der_update(
            preds, target, threshold=self.threshold
        )
        self.FalseAlarm += false_alarm
        self.MissedDetection += missed_detection
        self.SpeakerConfusion += speaker_confusion
        self.speech_total += speech_total

    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        opt_der, _ = torch.min(der, dim=0)

        return opt_der


class OptimalDiarizationErrorRateThreshold(OptimalDiarizationErrorRate):
    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        _, opt_threshold_idx = torch.min(der, dim=0)
        opt_threshold = self.threshold[opt_threshold_idx]

        return opt_threshold


class OptimalSpeakerConfusionRate(OptimalDiarizationErrorRate):
    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        _, opt_threshold_idx = torch.min(der, dim=0)
        return self.SpeakerConfusion[opt_threshold_idx] / self.speech_total


class OptimalFalseAlarmRate(OptimalDiarizationErrorRate):
    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        _, opt_threshold_idx = torch.min(der, dim=0)
        return self.FalseAlarm[opt_threshold_idx] / self.speech_total


class OptimalMissedDetectionRate(OptimalDiarizationErrorRate):
    def compute(self):
        der = _der_compute(
            self.FalseAlarm,
            self.MissedDetection,
            self.SpeakerConfusion,
            self.speech_total,
        )
        _, opt_threshold_idx = torch.min(der, dim=0)
        return self.MissedDetection[opt_threshold_idx] / self.speech_total
