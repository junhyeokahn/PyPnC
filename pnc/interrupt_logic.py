class InterruptLogic(object):
    def __init__(self):
        self._b_interrupt_button_eight = False

    def process_interrupts(self):
        self._reset_flags()

    def _reset_flags(self):
        self._b_interrupt_button_eight = False

    @property
    def b_interrupt_button_eight(self):
        return self._b_interrupt_button_eight

    @b_interrupt_button_eight.setter
    def b_interrupt_button_eight(self, value):
        self._b_interrupt_button_eight = value
