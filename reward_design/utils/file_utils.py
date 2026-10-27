import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict


def find_files_with_substring(directory, substring):
    matches = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file:
                matches.append(os.path.join(root, file))
    return matches


def load_tensorboard_logs(path):
    """Load all scalars from a TensorBoard log directory and its direct sub-directories."""
    data = defaultdict(list)
    dirs = [path] + [os.path.join(path, d) for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
    for log_dir in dirs:
        event_acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
        event_acc.Reload()  # Load all data written so far
        for tag in event_acc.Tags()["scalars"]:
            for event in event_acc.Scalars(tag):
                data[tag].append(event.value)

    return data


import importlib.util


def import_class_from_file(file_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, function_name)
    return function
