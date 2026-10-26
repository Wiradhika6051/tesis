
from types import SimpleNamespace

from src.localizer.GitDiffLocalizer import GitDiffLocalizer


def make_sample(diff, file_path="app.py", label=1):
    """Create a minimal sample for testing diff localization."""
    return SimpleNamespace(
        diff=diff,
        file_path=file_path,
        label=label,
    )


def test_vulnerable_sample_uses_deleted_lines():
    diff = """\
diff --git a/app.py b/app.py
index 1234567..abcdefg 100644
--- a/app.py
+++ b/app.py
@@ -1,5 +1,5 @@
 def get_user(user_id):
-    query = "SELECT * FROM users WHERE id = " + user_id
-    execute(query)
+    query = "SELECT * FROM users WHERE id = %s"
+    execute(query, (user_id,))
     return query
"""

    localizer = GitDiffLocalizer()

    sample = make_sample(diff, label=1)

    assert localizer.localize(sample) == [2, 3]


def test_fixed_sample_uses_added_lines():
    diff = """\
diff --git a/app.py b/app.py
index 1234567..abcdefg 100644
--- a/app.py
+++ b/app.py
@@ -1,5 +1,5 @@
 def get_user(user_id):
-    query = "SELECT * FROM users WHERE id = " + user_id
-    execute(query)
+    query = "SELECT * FROM users WHERE id = %s"
+    execute(query, (user_id,))
     return query
"""

    localizer = GitDiffLocalizer()

    sample = make_sample(diff, label=0)

    assert localizer.localize(sample) == [2, 3]


def test_only_target_file_contributes_seeds():
    diff = """\
diff --git a/app.py b/app.py
index 1234567..abcdefg 100644
--- a/app.py
+++ b/app.py
@@ -1,3 +1,3 @@
 def get_user(user_id):
-    query = build_query(user_id)
+    query = build_safe_query(user_id)
     return query
diff --git a/config.py b/config.py
index 1234567..abcdefg 100644
--- a/config.py
+++ b/config.py
@@ -1,2 +1,2 @@
-DEBUG = True
+DEBUG = False
"""

    localizer = GitDiffLocalizer()

    sample = make_sample(diff, file_path="app.py", label=1)

    assert localizer.localize(sample) == [2]


def test_multiple_hunks_in_target_file():
    diff = """\
diff --git a/app.py b/app.py
index 1234567..abcdefg 100644
--- a/app.py
+++ b/app.py
@@ -1,3 +1,3 @@
 def get_user(user_id):
-    query = build_query(user_id)
+    query = build_safe_query(user_id)
     return query
@@ -10,3 +10,3 @@
 def format_name(name):
-    return name.strip()
+    return name.strip().title()
"""

    localizer = GitDiffLocalizer()

    sample = make_sample(diff, file_path="app.py", label=1)

    assert localizer.localize(sample) == [2, 11]


def test_unrelated_function_changes_are_not_selected():
    diff = """\
diff --git a/app.py b/app.py
index 1234567..abcdefg 100644
--- a/app.py
+++ b/app.py
@@ -1,6 +1,6 @@
 def get_user(user_id):
-    query = build_query(user_id)
+    query = build_safe_query(user_id)
     return query

 def format_name(name):
-    return name.strip()
+    return name.strip().title()
"""

    localizer = GitDiffLocalizer()
    sample = make_sample(diff, file_path="app.py", label=1)

    assert localizer.localize(sample) == [2, 6]
