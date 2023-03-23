class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.
    """
    trainer = None  # A weak reference to the trainer object.

    def before_train(self):
        pass

    def before_epoch(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def after_epoch(self):
        pass

    def after_train(self):
        pass
