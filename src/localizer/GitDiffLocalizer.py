import re

from tesis.dataset.sample import Sample


class GitDiffLocalizer:

    HUNK_PATTERN = re.compile(
        r'^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@'
    )

    def extract(self, sample: Sample):

        seed_lines = []

        old_line = None
        new_line = None

        for line in sample.diff.splitlines():

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
            old_line += 1
            new_line += 1

        return seed_lines