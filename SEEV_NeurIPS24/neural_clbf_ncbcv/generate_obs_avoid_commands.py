import itertools


def generate_commands(script_name, option_dict):
    # Get all combinations of the option values
    keys, values = zip(*option_dict.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Generate the commands
    commands = []
    for combination in combinations:
        cmd = f"python {script_name}"

        for key, value in combination.items():
            if isinstance(value, bool):
                if value:
                    cmd += f" --{key}"
                else:
                    cmd += f" --no-{key}"
            else:
                cmd += f" --{key} {value}"
        commands.append(cmd)

    return commands


# Example usage
script_name = "neural_clbf/training/train_obs_avoid.py"
option_dict = {
    "cbf_hidden_layers": [2, 4, 8],
    "cbf_hidden_size": [8, 16],
    "random_seed": [111, 222, 333],
    "perform_certification": [True],
    "sim_reg_weight": [0.0, 1.0, 10.0, 50.0],
}

commands = generate_commands(script_name, option_dict)
with open("obs_avoid_commands.txt", "w") as f:
    for cmd in commands:
        f.write(f"{cmd}\n")
