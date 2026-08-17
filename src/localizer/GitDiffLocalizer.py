import re

from pyparsing import line

from src.localizer.DiffLocalizer import DiffLocalizer
from src.type.Sample import Sample


class GitDiffLocalizer(DiffLocalizer):

    HUNK_PATTERN = re.compile(
        r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@'
    )

    def localize(self, sample: Sample):
        diff = self.get_file_diff(sample.diff, sample.file_path)
        seed_lines = []

        old_line = None
        new_line = None

        for line in diff.splitlines():

            #
            # New hunk
            #
            match = self.HUNK_PATTERN.match(line)

            if match:

                old_line = int(match.group(1))
                new_line = int(match.group(2))

                continue

            #
            # Skip until first hunk
            #
            if old_line is None:
                continue

            #
            # Deleted line
            #
            if line.startswith("-") and not line.startswith("---"):

                if sample.label == 1:
                    seed_lines.append(old_line)

                old_line += 1
                continue

            #
            # Added line
            #
            if line.startswith("+") and not line.startswith("+++"):

                if sample.label == 0:
                    seed_lines.append(new_line)

                new_line += 1
                continue

            #
            # Context line
            #
            if line.startswith("\\"):
                continue
            
            old_line += 1
            new_line += 1

        return seed_lines

    def get_file_diff(self,diff, target_file):
        current_file = None
        lines = []

        for line in diff.splitlines():

            if line.startswith("diff --git "):
                match = re.match(r"diff --git a/(.*?) b/(.*)", line)

                if match:
                    current_file = match.group(2)

            if current_file == target_file:
                lines.append(line)

        return "\n".join(lines)