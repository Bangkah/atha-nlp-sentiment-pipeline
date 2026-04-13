# Contributing

Terima kasih sudah ingin berkontribusi.

## Workflow

1. Fork repository.
2. Buat branch fitur baru dari `main`.
3. Jalankan quality check lokal:

```bash
make lint
make test
```

4. Pastikan perubahan terdokumentasi jika memengaruhi behavior.
5. Kirim Pull Request dengan deskripsi jelas.

## Branch Protection (Recommended)

Untuk menjaga kualitas branch `main`, aktifkan branch protection di GitHub dengan aturan berikut:

- Require a pull request before merging.
- Require status checks to pass before merging.
- Require branches to be up to date before merging.
- Do not allow force pushes.

Status checks yang direkomendasikan sebagai required:

- `quality-check`
- `docker-build`
- `docker-smoke`

Kebijakan merge:

- PR hanya boleh di-merge saat semua status check di atas hijau.
- Jika ada check gagal, perbaiki dulu sampai lulus sebelum merge.

## Coding Notes

- Gunakan Python 3.11+.
- Untuk data training custom, gunakan kolom CSV `text,label`.
- Label valid: `0=negative`, `1=neutral`, `2=positive`.

## Pull Request Checklist

- [ ] Kode lolos `make lint`
- [ ] Kode lolos `make test`
- [ ] Semua GitHub Actions check hijau (`quality-check`, `docker-build`, `docker-smoke`)
- [ ] README atau docs sudah diperbarui bila relevan
- [ ] Tidak ada secret/token yang ikut ter-commit
