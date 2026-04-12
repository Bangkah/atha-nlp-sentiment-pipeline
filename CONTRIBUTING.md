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

## Coding Notes

- Gunakan Python 3.11+.
- Untuk data training custom, gunakan kolom CSV `text,label`.
- Label valid: `0=negative`, `1=neutral`, `2=positive`.

## Pull Request Checklist

- [ ] Kode lolos `make lint`
- [ ] Kode lolos `make test`
- [ ] README atau docs sudah diperbarui bila relevan
- [ ] Tidak ada secret/token yang ikut ter-commit
