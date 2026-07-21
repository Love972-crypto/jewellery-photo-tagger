from urllib.request import urlopen

from src.download_server import download_url, start_download_server


def test_parallel_download_servers_keep_runtime_roots_isolated(tmp_path):
    project_a = tmp_path / "project_a"
    project_b = tmp_path / "project_b"
    archive_a = project_a / ".runtime" / "run_1" / "Jewellery_Output" / "processed_images.zip"
    archive_b = project_b / ".runtime" / "run_1" / "Jewellery_Output" / "processed_images.zip"
    archive_a.parent.mkdir(parents=True)
    archive_b.parent.mkdir(parents=True)
    archive_a.write_bytes(b"project-a-zip")
    archive_b.write_bytes(b"project-b-zip")

    server_a = start_download_server(project_a)
    server_b = start_download_server(project_b)
    try:
        assert server_a.base_url != server_b.base_url

        with urlopen(download_url(server_a, archive_a), timeout=5) as response:
            assert response.status == 200
            assert response.read() == b"project-a-zip"
            assert response.headers["Content-Disposition"] == 'attachment; filename="processed_images.zip"'

        with urlopen(download_url(server_b, archive_b), timeout=5) as response:
            assert response.status == 200
            assert response.read() == b"project-b-zip"
    finally:
        server_a.close()
        server_b.close()
