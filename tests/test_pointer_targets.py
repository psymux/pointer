import unittest

from pointer_targets import TargetResolver, _parse_tle_triplets, parse_horizons_ephemeris


HORIZONS_SAMPLE = """
header
$$SOE
 2026-Mar-08 00:00 *   17 17 02.22 +12 11 32.1  285.137489  -0.159469  169.887314323989  -7.2797502
 2026-Mar-08 00:10 *s  17 17 02.22 +12 11 32.2  283.643841  -2.102533  169.887285132930  -7.2775856
$$EOE
footer
"""

HORIZONS_SAMPLE_WITH_M_FLAG = """
header
$$SOE
 2026-Mar-07 16:00  m  10 54 04.47 +09 44 18.6  299.374567  19.756292  0.01329888048023   0.3383738
$$EOE
footer
"""


class ParseHelpersTest(unittest.TestCase):
    def test_parse_tle_triplets(self):
        payload = "\n".join(
            [
                "ISS (ZARYA)",
                "1 25544U 98067A   26067.12345678  .00012345  00000+0  12345-3 0  9990",
                "2 25544  51.6435 123.4567 0001234 123.4567 234.5678 15.50000000123456",
                "HST",
                "1 20580U 90037B   26067.12345678  .00012345  00000+0  12345-3 0  9990",
                "2 20580  28.4690 123.4567 0001234 123.4567 234.5678 15.50000000123456",
            ]
        )
        triplets = _parse_tle_triplets(payload)
        self.assertEqual(2, len(triplets))
        self.assertEqual("ISS (ZARYA)", triplets[0]["name"])
        self.assertEqual("HST", triplets[1]["name"])

    def test_parse_horizons_ephemeris(self):
        samples = parse_horizons_ephemeris(HORIZONS_SAMPLE)
        self.assertEqual(2, len(samples))
        self.assertAlmostEqual(285.137489, samples[0]["az_deg"])
        self.assertAlmostEqual(-2.102533, samples[1]["alt_deg"])
        self.assertAlmostEqual(259.25925, samples[0]["ra_deg"], places=3)
        self.assertAlmostEqual(12.19225, samples[0]["dec_deg"], places=3)

    def test_parse_horizons_ephemeris_with_visibility_flag(self):
        samples = parse_horizons_ephemeris(HORIZONS_SAMPLE_WITH_M_FLAG)
        self.assertEqual(1, len(samples))
        self.assertAlmostEqual(163.518625, samples[0]["ra_deg"], places=3)
        self.assertAlmostEqual(9.7385, samples[0]["dec_deg"], places=3)
        self.assertAlmostEqual(299.374567, samples[0]["az_deg"])


class LocalCatalogSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolver = TargetResolver()

    def test_search_solar_system(self):
        matches = self.resolver.search("mars", kind="solar-system")
        self.assertTrue(matches)
        self.assertEqual("solar-system", matches[0].kind)

    def test_search_constellation(self):
        matches = self.resolver.search("orion", kind="constellation")
        self.assertTrue(matches)
        self.assertEqual("Ori", matches[0].identifier)

    def test_search_deep_sky(self):
        matches = self.resolver.search("andromeda", kind="dso")
        self.assertTrue(matches)
        self.assertEqual("Andromeda Galaxy", matches[0].display_name)


if __name__ == "__main__":
    unittest.main()
