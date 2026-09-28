"""Utterance legality and sequence enumeration."""

def is_legal(utterance_map, utterance_history):

    remaining_time = 0
    tot_cost = 0

    for utt in utterance_history:
        try:
            row = utterance_map.loc[utt]
        except KeyError:
            return 0, tot_cost

        time_required = row["Time"]
        blocks_time = row["Usage"]
        cost = row["Cost"]

        remaining_time += 1
        if time_required > remaining_time:
            tot_cost += cost
            return 0, tot_cost # because it breaks out costs bot updated for leftover seq

        if time_required <= remaining_time:
            # blocked = blocks_time == 1
            tot_cost += cost
            if blocks_time:
                remaining_time = 0

    return 1, tot_cost

def generate_legal_sequences(utterance_map, sequence_length):
    legal_sequences = []

    def traverse(current_state, current_sequence):
        if len(current_sequence) == sequence_length:
            legal_sequences.append(current_sequence)
            return

        for utterance, row in utterance_map.iterrows():
            time_required = row["Time"]
            blocks_time = row["Usage"]

            if time_required > current_state + 1:
                continue

            next_state = 0 if blocks_time else current_state + 1
            traverse(next_state, current_sequence + [utterance])

    traverse(0, [])
    return legal_sequences

def is_legal_reward_end(utterance_map, utterance_history):

    required_blocks = 0
    total_cost = 0
    for idx, message in enumerate(utterance_history):
        # Check if message exists in the map
        if message not in utterance_map.index:
            return False, 0
        # Check if current message is Block when required
        if required_blocks > 0:
            if message != '(XXX)':
                return False, 0
            required_blocks -= 1
            total_cost += utterance_map.loc[message, 'Cost']
        else:
            if message == '(XXX)':
                return False, 0
            # Get message properties
            time = utterance_map.loc[message, 'Time']
            # Check if there are enough steps left for this message
            if idx + time > len(utterance_history):
                return False, 0
            required_blocks = time - 1
            total_cost += utterance_map.loc[message, 'Cost']
    return True, total_cost

def generate_legal_sequences_re(utterance_map, sequence_length):
    legal_sequences = []

    def backtrack(current_seq, current_pos, required_blocks):
        if current_pos == sequence_length:
            legal_sequences.append(current_seq.copy())
            return

        if required_blocks > 0:
            # The next must be (XXX)
            current_seq.append('(XXX)')
            backtrack(current_seq, current_pos + 1, required_blocks - 1)
            current_seq.pop()
        else:
            # Choose any message except (XXX) that fits
            for message in utterance_map.index:
                if message == '(XXX)':
                    continue
                time = utterance_map.loc[message, 'Time']
                if current_pos + time > sequence_length:
                    continue
                current_seq.append(message)
                backtrack(current_seq, current_pos + 1, time - 1)
                current_seq.pop()

    backtrack([], 0, 0)
    return legal_sequences
