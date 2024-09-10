import numpy as np
import torch


class SamplingSchemeBase:
    def __init__(
        self,
        video_length: int,
        num_observed_frames: int,
        max_frames: int,
        step_size: int,
    ):
        """Sampling scheme base class. It provides an iterator that returns
            the indices of the frames that should be observed and the frames that should be generated.

        Args:
            video_length (int): Length of the videos.
            num_obs (int): Number of frames that are observed from the beginning of the video.
            max_frames (int): Maximum number of frames (observed or latent) that can be passed to the model in one shot.
            step_size (int): Number of frames to generate in each step.
        """
        print(f'Inferring using the sampling scheme "{self.typename}".')
        self._video_length = video_length
        self._max_frames = max_frames
        self._num_obs = num_observed_frames
        self._done_frames = set(range(self._num_obs))
        self._obs_frames = list(range(self._num_obs))
        self._step_size = step_size
        self._current_step = 0  # Counts the number of steps.
        self.B = None

    def get_unconditional_indices(self):
        return list(range(self._max_frames))

    def __next__(self):
        # Check if the video is fully generated.
        if self.is_done():
            raise StopIteration
        unconditional = False
        if self._num_obs == 0 and self._current_step == 0:
            # Handles unconditional sampling by sampling a batch of self._max_frame latent frames in the first
            # step, then proceeding as usual in the next steps.
            obs_frame_indices = []
            latent_frame_indices = self.get_unconditional_indices()
            unconditional = True
        else:
            # Get the next indices from the function overloaded by each sampling scheme.
            obs_frame_indices, latent_frame_indices = self.next_indices()

        # Type checks. Both observed and latent indices should be lists.
        assert isinstance(obs_frame_indices, list) and isinstance(
            latent_frame_indices, list
        )
        # Make sure the observed frames are either osbserved or already generated before
        for idx in obs_frame_indices:
            assert (
                idx in self._done_frames
            ), f"Attempting to condition on frame {idx} while it is not generated yet.\nGenerated frames: {self._done_frames}\nObserving: {obs_frame_indices}\nGenerating: {latent_frame_indices}"
        assert np.all(np.array(latent_frame_indices) < self._video_length)
        self._done_frames.update(
            [idx for idx in latent_frame_indices if idx not in self._done_frames]
        )
        if unconditional:
            # Allows the unconditional sampling to continue the next sampling stages as if it was a conditional model.
            self._obs_frames = latent_frame_indices
        self._current_step += 1
        if self.B is not None:
            obs_frame_indices = [obs_frame_indices] * self.B
            latent_frame_indices = [latent_frame_indices] * self.B

        # Generate the boolean temporal frame masks. True indicates latent frames,
        # False indicates observed frames, of shape (B,T)
        temporal_mask = torch.ones(
            (len(obs_frame_indices), self._max_frames), dtype=torch.bool
        )
        for batch_idx in range(len(obs_frame_indices)):
            for frame_idx in obs_frame_indices[batch_idx]:
                # Convert from absolute frame index to relative
                frame_idx = frame_idx - (self._step_size) * (self._current_step - 1)

                assert frame_idx >= 0 and frame_idx < self._max_frames
                temporal_mask[batch_idx][frame_idx] = False
        return obs_frame_indices, latent_frame_indices, temporal_mask

    def is_done(self):
        return len(self._done_frames) >= self._video_length

    def __iter__(self):
        self.step = 0
        return self

    def next_indices(self):
        raise NotImplementedError

    @property
    def typename(self):
        return type(self).__name__

    def set_videos(self, videos):
        self.B = len(videos)

    @property
    def num_observations(self):
        return self._num_obs

    @property
    def video_length(self):
        return self._video_length


class Autoregressive(SamplingSchemeBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def next_indices(self):
        if len(self._done_frames) == 0:
            return [], list(range(self._max_frames))
        obs_frame_indices = sorted(self._done_frames)[
            -(self._max_frames - self._step_size) :
        ]
        first_idx = obs_frame_indices[-1] + 1
        latent_frame_indices = list(
            range(first_idx, min(first_idx + self._step_size, self._video_length))
        )

        return obs_frame_indices, latent_frame_indices
