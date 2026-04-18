import os
import tempfile
import unittest

from src.memory.profile_store import ProfileStore


class ProfileStoreTests(unittest.TestCase):
    def test_profile_crud_and_active_profile_persistence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = os.path.join(temp_dir, "profiles.json")
            store = ProfileStore(storage_path=storage)

            p1 = store.create_profile(
                profile_name="Main",
                questionnaire_answers={"Gender_Male": 1.0},
                model_artifacts_path="C:/models",
            )
            self.assertEqual(store.get_active_profile().profile_id, p1.profile_id)

            p2 = store.create_profile(profile_name="Secondary")
            self.assertEqual(len(store.list_profiles()), 2)

            updated = store.update_profile(
                p2.profile_id,
                profile_name="Secondary Updated",
                export_preferences={"format": "csv"},
            )
            self.assertEqual(updated.profile_name, "Secondary Updated")
            self.assertEqual(updated.export_preferences["format"], "csv")

            store.set_active_profile(p2.profile_id)
            self.assertEqual(store.get_active_profile().profile_id, p2.profile_id)

            # New instance should load persisted data.
            reloaded = ProfileStore(storage_path=storage)
            self.assertEqual(len(reloaded.list_profiles()), 2)
            self.assertEqual(reloaded.get_active_profile().profile_id, p2.profile_id)

            reloaded.delete_profile(p2.profile_id)
            self.assertEqual(len(reloaded.list_profiles()), 1)
            self.assertEqual(reloaded.get_active_profile().profile_id, p1.profile_id)

            with self.assertRaises(KeyError):
                reloaded.delete_profile("missing")


if __name__ == "__main__":
    unittest.main()

