import os
import pickle


class CheckpointManager:

    def __init__(
        self,
        checkpoint_dir
    ):

        self.checkpoint_dir = checkpoint_dir

        os.makedirs(
            checkpoint_dir,
            exist_ok=True
        )

    def get_repo_checkpoint(
        self,
        repo_name
    ):

        return os.path.join(
            self.checkpoint_dir,
            f"{repo_name}.pkl"
        )

    def repository_finished(
        self,
        repo_name
    ):

        return os.path.exists(
            self.get_repo_checkpoint(
                repo_name
            )
        )

    def load_repository(
        self,
        repo_name
    ):

        path = self.get_repo_checkpoint(
            repo_name
        )

        if not os.path.exists(
            path
        ):

            return {}

        with open(
            path,
            "rb"
        ) as f:

            return pickle.load(
                f
            )

    def save_repository(
        self,
        repo_name,
        lookup
    ):

        path = self.get_repo_checkpoint(
            repo_name
        )

        with open(
            path,
            "wb"
        ) as f:

            pickle.dump(
                lookup,
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    def merge(
        self,
        output_file
    ):

        merged = {}

        for filename in os.listdir(
            self.checkpoint_dir
        ):

            if not filename.endswith(
                ".pkl"
            ):

                continue

            path = os.path.join(
                self.checkpoint_dir,
                filename
            )

            with open(
                path,
                "rb"
            ) as f:

                merged.update(
                    pickle.load(f)
                )

        with open(
            output_file,
            "wb"
        ) as f:

            pickle.dump(
                merged,
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        return merged

    def load_all(self):

        merged = {}

        for filename in os.listdir(
            self.checkpoint_dir
        ):

            if filename.endswith(
                ".pkl"
            ):

                path = os.path.join(
                    self.checkpoint_dir,
                    filename
                )

                with open(
                    path,
                    "rb"
                ) as f:

                    merged.update(
                        pickle.load(f)
                    )

        return merged

    def clear(self):

        for filename in os.listdir(
            self.checkpoint_dir
        ):

            if filename.endswith(
                ".pkl"
            ):

                os.remove(
                    os.path.join(
                        self.checkpoint_dir,
                        filename
                    )
                )