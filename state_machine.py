"""
state_machine.py - SquadMind Finite State Machine (FSM)
Tracks in-game state transitions with hysteresis and debounce protection.
"""

import time
import logging
from collections import deque

logger = logging.getLogger("StateMachine")


class GameStateMachine:
    """Manages high-level gameplay context (e.g. In-Game, Menu, Fastbreak, Timeout)
    and dispatches tactical events to controllers and audio engines.
    """
    def __init__(self, debounce_duration=0.6, confidence_frames=3):
        self.current_state = "IDLE"
        self.candidate_state = "IDLE"
        self.candidate_count = 0
        self.confidence_frames = confidence_frames
        self.debounce_duration = debounce_duration
        self.last_transition_time = time.time()

        self.callbacks = {}
        self.history = deque(maxlen=20)

    def register_callback(self, trigger_state, func):
        """Registers a function to be executed when entering `trigger_state`."""
        if trigger_state not in self.callbacks:
            self.callbacks[trigger_state] = []
        self.callbacks[trigger_state].append(func)

    def update(self, analysis_result):
        """Evaluates latest vision analysis result and computes state transitions."""
        now = time.time()
        in_game = analysis_result.get("in_game", False)
        hud_states = analysis_result.get("hud_states", {})
        stamina = hud_states.get("stamina_percentage", 0.0)

        # 1. Determine raw candidate state
        if not in_game:
            target = "MENU_OR_LOBBY"
        else:
            if stamina > 0.6:
                target = "ACTIVE_GAMEPLAY_HIGH_STAMINA"
            elif stamina > 0.0:
                target = "ACTIVE_GAMEPLAY_LOW_STAMINA"
            else:
                target = "ACTIVE_GAMEPLAY"

        # 2. Hysteresis check (require consecutive consistent detections)
        if target == self.candidate_state:
            self.candidate_count += 1
        else:
            self.candidate_state = target
            self.candidate_count = 1

        # 3. Transition if confidence threshold is met and debounce window passed
        if (self.candidate_count >= self.confidence_frames and 
            self.candidate_state != self.current_state and 
            (now - self.last_transition_time) >= self.debounce_duration):

            old_state = self.current_state
            self.current_state = self.candidate_state
            self.last_transition_time = now

            transition_event = {
                "timestamp": now,
                "from_state": old_state,
                "to_state": self.current_state,
                "stamina": stamina
            }
            self.history.append(transition_event)
            logger.info(f"FSM State Transition: {old_state} -> {self.current_state}")

            # Fire registered callbacks
            if self.current_state in self.callbacks:
                for cb in self.callbacks[self.current_state]:
                    try:
                        cb(transition_event)
                    except Exception as e:
                        logger.error(f"Callback error in state {self.current_state}: {e}")

        return self.current_state
