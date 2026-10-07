---
paths:
  - "frontend/**/*"
---

# Frontend rules (stage 2)

- Every visible string is an i18n key that exists in both `public/locales/es/translation.json`
  and `public/locales/en/translation.json`. No hard-coded UI text.
- Data only through the API in `backend/openapi.json`; no direct file reads.
- Map layers with many polygons use the canvas renderer.
- Months with `quality = no_data` are shown greyed out and never interpolated silently.
