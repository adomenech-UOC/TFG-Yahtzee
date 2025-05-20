from tensor_encoder import encode_state
import numpy as np
from collections import OrderedDict

YAHTZEE_CATEGORIES = [
    'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes',
    'Three of a Kind', 'Four of a Kind', 'Full House',
    'Small Straight', 'Large Straight', 'Yahtzee', 'Chance'
]

def generate_all_actions():
    actions = []
    
    # 1) All possible keep masks: 2^5 = 32
    for mask_int in range(32):
        keep_mask = [(mask_int & (1 << i)) != 0 for i in range(5)]
        # keep_mask is a list of T/F for each die
        actions.append(('reroll', keep_mask))
    
    # 2) All possible scoring categories (13)
    for cat in YAHTZEE_CATEGORIES:
        actions.append(('score', cat))

    return actions

ALL_ACTIONS = generate_all_actions()  # length = 32 + 13 = 45

class YahtzeeGame:
    def __init__(self):
        self.categories = OrderedDict([
            ('Ones', None), ('Twos', None), ('Threes', None), ('Fours', None),
            ('Fives', None), ('Sixes', None), ('Three of a Kind', None),
            ('Four of a Kind', None), ('Full House', None),
            ('Small Straight', None), ('Large Straight', None),
            ('Yahtzee', None), ('Chance', None)
        ])

        self.upper_bonus = 0
        self.yahtzee_bonuses = 0
        self.dice = [0]*5
        self.roll_dice()
        
        # Start with 0 rolls_left; it will be set to 3 internally
        # whenever the first reroll is requested each turn:
        self.rolls_left = 0

    def reset(self):
        for cat in self.categories:
            self.categories[cat] = None
        self.upper_bonus = 0
        self.yahtzee_bonuses = 0
        self.dice = [0]*5
        self.roll_dice()
        self.rolls_left = 2
        return self.get_encoded_state()

    def get_state(self):
        return {
            'categories': self.categories.copy(),
            'dice': self.dice.copy(),
            'rolls_left': self.rolls_left,
            'upper_bonus': self.upper_bonus,
            'yahtzee_bonuses': self.yahtzee_bonuses
        }

    def get_encoded_state(self):
        return encode_state(self.get_state())

    def set_state(self, state):
        self.categories = state['categories'].copy()
        self.dice = state['dice'].copy()
        self.rolls_left = state['rolls_left']
        self.upper_bonus = state['upper_bonus']
        self.yahtzee_bonuses = state['yahtzee_bonuses']

    def roll_dice(self, keep_mask=None):
        if keep_mask is None:
            # Generate all 5 dice at once
            self.dice = np.random.randint(1, 7, size=5)
        else:
            # Generate new values only for non-kept dice
            new_values = np.random.randint(1, 7, size=5)
            self.dice = np.where(keep_mask, self.dice, new_values)

        self.dice.sort()

    def get_possible_moves(self):
        return [cat for cat, score in self.categories.items() if score is None]

    def calculate_score(self, category, dice):
        dice_np = np.array(dice)
        counts = np.bincount(dice_np, minlength=7)[
            1:]  # Count occurrences of 1-6
        number_map = {
            'Ones':   1,
            'Twos':   2,
            'Threes': 3,
            'Fours':  4,
            'Fives':  5,
            'Sixes':  6
        }
        # joker rule flag: bonus yahtzee can be scored as anything
        bonus_yahtzee = all(
            d == self.dice[0] for d in self.dice) and self.categories['Yahtzee'] == 50

        if category in number_map:
            value = number_map[category]
            return sum(d for d in dice if d == value)
        elif category == 'Three of a Kind':
            return sum(dice) if max(counts) >= 3 else 0
        elif category == 'Four of a Kind':
            return sum(dice) if max(counts) >= 4 else 0
        elif category == 'Full House':
            return 25 if (3 in counts and 2 in counts) or bonus_yahtzee else 0
        elif category == 'Small Straight':
            return 30 if any(all(x in dice for x in [i, i+1, i+2, i+3]) for i in [1, 2, 3]) or bonus_yahtzee else 0
        elif category == 'Large Straight':
            return 40 if any(all(x in dice for x in [i, i+1, i+2, i+3, i+4]) for i in [1, 2]) or bonus_yahtzee else 0
        elif category == 'Yahtzee':
            return 50 if all(d == dice[0] for d in dice) else 0
        elif category == 'Chance':
            return sum(dice)
        return 0

    def apply_move(self, category):

        score = self.calculate_score(category, self.dice)
        bonuses = 0

        # Check if we're crossing 63 in the upper section for the first time
        if category in ['Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes']:
            upper_section = list(self.categories.items())[:6]
            upper_total = sum(v for k, v in upper_section if v is not None)
            if upper_total + score >= 63 and self.upper_bonus == 0:
                self.upper_bonus = 35
                bonuses += 35

        # Extra Yahtzee bonus if we've already scored Yahtzee category previously
        if category == 'Yahtzee' and score == 50 and self.categories['Yahtzee'] is not None:
            self.yahtzee_bonuses += 100
            bonuses += 100

        self.categories[category] = score
        return score, bonuses

    def get_total_score(self):
        final_score = sum(v for v in self.categories.values() if v is not None)
        final_score += self.upper_bonus + self.yahtzee_bonuses
        return final_score

    def step(self, action):
        """
        action is either:
          ('reroll', keep_mask)
          ('score', category_name)
        """
        reward = 0.0
        done = False

        if action[0] == 'reroll':
            keep_mask = action[1]
            # If rolls_left == 0, treat this as the start of a brand-new turn
            if self.rolls_left == 0:
                self.rolls_left = 2

            if self.rolls_left > 0:
                self.roll_dice(keep_mask)
                self.rolls_left -= 1  # use up one roll

        elif action[0] == 'score':
            category = action[1]
            if self.categories[category] is not None:
                # Category used already
                reward = 0
            else:
                score, bonuses = self.apply_move(category)
                reward = score + bonuses

            # Scoring ends the turn; set rolls_left to 3
            self.roll_dice()
            self.rolls_left = 2

            # Check if game ended (all categories filled)
            if all(v is not None for v in self.categories.values()):
                reward += self.get_total_score()
                done = True

        next_state = self.get_encoded_state()
        return next_state, reward, done, {}

    def get_valid_actions_mask(self):
        """Returns boolean mask for all 45 actions (32 reroll patterns + 13 categories)"""
        mask = [False] * 45

        # if at the beginning of a turn, must roll all dice.

        if self.rolls_left > 0:
            # Rolling phase: can reroll (first 32 actions)
            mask[:32] = [True] * 32
        else:
            # Scoring phase: can choose unused categories
            available_cats = [i for i, (cat, score) in enumerate(self.categories.items())
                              if score is None]
            for idx in available_cats:
                mask[32 + idx] = True  # Scoring actions start at index 32

        return mask
