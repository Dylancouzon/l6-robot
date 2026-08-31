import ipaddress
import tempfile
import unittest
from pathlib import Path

from cryptography import x509

from robot.device.server import ensure_cert


class CertificateTests(unittest.TestCase):
    def test_certificate_is_generated_without_an_openssl_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            cert_path, key_path = ensure_cert("192.0.2.10", root=directory)

            self.assertTrue(Path(key_path).read_text().startswith(
                "-----BEGIN RSA PRIVATE KEY-----"))
            certificate = x509.load_pem_x509_certificate(
                Path(cert_path).read_bytes())
            san = certificate.extensions.get_extension_for_class(
                x509.SubjectAlternativeName).value
            self.assertIn(ipaddress.ip_address("192.0.2.10"),
                          san.get_values_for_type(x509.IPAddress))
            self.assertIn(ipaddress.ip_address("127.0.0.1"),
                          san.get_values_for_type(x509.IPAddress))
            self.assertIn("localhost", san.get_values_for_type(x509.DNSName))

            # A second call for the same names reuses the trusted certificate.
            first = Path(cert_path).read_bytes()
            self.assertEqual(ensure_cert("192.0.2.10", root=directory),
                             (cert_path, key_path))
            self.assertEqual(Path(cert_path).read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
