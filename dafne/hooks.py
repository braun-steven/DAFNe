from detectron2.engine.train_loop import HookBase
from dafne.utils.rtpt import RTPT


class RTPTHook(HookBase):
    def before_train(self):
        """
        Called before the first iteration.
        """
        # High moving avg size due to large differences in number of objects per image
        window_size = 500
        self.rtpt = RTPT(
            name_initials="SL",
            experiment_name=self.trainer.cfg.EXPERIMENT_NAME,
            max_iterations=self.trainer.max_iter,
            iteration_start=self.trainer.start_iter,
            update_interval=100,
            moving_avg_window_size=window_size,
        )
        self.rtpt.start()

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        progress = self.trainer.iter / self.trainer.max_iter * 100
        self.rtpt.step(subtitle=f"[{progress:0>2.0f}%]")
        pass
