import json
import time


class Statistics:

    def __init__(self):

        self.start_time = time.time()

        self.stats = {

            #
            # Repository
            #
            "repositories": 0,
            "repositories_skipped": 0,
            "repositories_failed": 0,

            #
            # Commit
            #
            "commits": 0,
            "commit_failures": 0,

            #
            # File
            #
            "files": 0,
            "files_success": 0,
            "files_missing": 0,
            "file_failures": 0,

            #
            # Git
            #
            "clone_failures": 0,

            #
            # Cache
            #
            "parent_cache_hit": 0,
            "parent_cache_miss": 0,

            "file_cache_hit": 0,
            "file_cache_miss": 0

        }

    def increment(
        self,
        key,
        amount=1
    ):

        self.stats[key] += amount

    def finish(self):

        self.stats[
            "elapsed_seconds"
        ] = round(
            time.time() - self.start_time,
            2
        )

    def report(self):

        self.finish()

        s = self.stats

        #
        # Derived statistics
        #

        if s["repositories"] > 0:

            s["avg_commits_per_repo"] = round(

                s["commits"] /
                s["repositories"],

                2
            )

            s["avg_files_per_repo"] = round(

                s["files"] /
                s["repositories"],

                2
            )

        else:

            s["avg_commits_per_repo"] = 0

            s["avg_files_per_repo"] = 0

        if s["files"] > 0:

            s["success_rate"] = round(

                s["files_success"] /
                s["files"],

                4
            )

            s["missing_rate"] = round(

                s["files_missing"] /
                s["files"],

                4
            )

        else:

            s["success_rate"] = 0

            s["missing_rate"] = 0

        total_parent = (

            s["parent_cache_hit"] +

            s["parent_cache_miss"]

        )

        if total_parent:

            s["parent_cache_hit_rate"] = round(

                s["parent_cache_hit"] /
                total_parent,

                4
            )

        else:

            s["parent_cache_hit_rate"] = 0

        total_file = (

            s["file_cache_hit"] +

            s["file_cache_miss"]

        )

        if total_file:

            s["file_cache_hit_rate"] = round(

                s["file_cache_hit"] /
                total_file,

                4
            )

        else:

            s["file_cache_hit_rate"] = 0

        return s

    def save(
        self,
        filename
    ):

        with open(
            filename,
            "w"
        ) as f:

            json.dump(

                self.report(),

                f,

                indent=4

            )

    def print_report(self):

        report = self.report()

        print()

        print("=" * 60)

        print("Fetch Statistics")

        print("=" * 60)

        for key, value in report.items():

            print(

                f"{key:30}: {value}"

            )