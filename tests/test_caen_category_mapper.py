import unittest

from src.domain.enums import CategoryArea
from src.features.caen_category_mapper import (
    choose_primary_expense_category,
    map_caen_to_expense_category,
    map_caen_to_impulse_buying_category,
    map_caen_to_product_lifetime_bucket,
)


class CaenCategoryMapperTests(unittest.TestCase):
    def test_expense_category_mapping_from_caen(self):
        self.assertEqual(map_caen_to_expense_category("5610"), CategoryArea.FOOD.value)
        self.assertEqual(map_caen_to_expense_category("3515"), CategoryArea.HOUSING.value)
        self.assertEqual(map_caen_to_expense_category("4932"), CategoryArea.TRANSPORT.value)
        self.assertEqual(map_caen_to_expense_category("5914"), CategoryArea.ENTERTAINMENT.value)
        self.assertEqual(map_caen_to_expense_category("4773"), CategoryArea.HEALTH.value)
        self.assertEqual(map_caen_to_expense_category("9602"), CategoryArea.PERSONAL_CARE.value)
        self.assertEqual(map_caen_to_expense_category("8559"), CategoryArea.CHILD_EDUCATION.value)

    def test_impulse_category_mapping_from_caen(self):
        self.assertEqual(map_caen_to_impulse_buying_category("4741"), "Electronics or gadgets")
        self.assertEqual(map_caen_to_impulse_buying_category("4771"), "Clothing or personal care products")
        self.assertEqual(map_caen_to_impulse_buying_category("5914"), "Entertainment")
        self.assertEqual(map_caen_to_impulse_buying_category("5610"), "Food")
        self.assertEqual(map_caen_to_impulse_buying_category("7022"), "Other")

    def test_product_lifetime_bucket_mapping_from_caen(self):
        self.assertEqual(map_caen_to_product_lifetime_bucket("4742"), "Tech")
        self.assertEqual(map_caen_to_product_lifetime_bucket("4754"), "Appliances")
        self.assertEqual(map_caen_to_product_lifetime_bucket("4511"), "Cars")
        self.assertEqual(map_caen_to_product_lifetime_bucket("4772"), "Clothing")
        self.assertIsNone(map_caen_to_product_lifetime_bucket("5610"))

    def test_choose_primary_category_for_multi_caen_company(self):
        category = choose_primary_expense_category(["5610", "4711", "7022"])
        self.assertEqual(category, CategoryArea.FOOD.value)

    def test_handles_invalid_codes(self):
        self.assertIsNone(map_caen_to_expense_category(None))
        self.assertIsNone(map_caen_to_expense_category(""))
        self.assertEqual(map_caen_to_impulse_buying_category(None), "Other")
        self.assertIsNone(choose_primary_expense_category([]))


if __name__ == "__main__":
    unittest.main()

