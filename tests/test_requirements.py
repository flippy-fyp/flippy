from lib.constants import REPO_ROOT
import unittest
import os
import pkg_resources


REQUIREMENTS_TXT_PATH = os.path.join(REPO_ROOT, "requirements.txt")


class TestRequirements(unittest.TestCase):
    def test_no_nsgt_in_requirements(self):
        """
        nsgt cannot be in requirements.txt and must be installed separately AFTER
        installing all requirements in requirements.txt
        """
        with open(REQUIREMENTS_TXT_PATH, "r") as f:
            contents = f.read()
            self.assertEqual(
                contents.find("nsgt"),
                -1,
                "Please remove nsgt entry from requirements.txt--"
                + "it must be installed separately after installing all requirements in requirements.txt",
            )

    def test_all_requirements_ok(self):
        with open(REQUIREMENTS_TXT_PATH, "r") as f:
            dependencies = f.read().split("\n")
            pkg_resources.require(dependencies)
