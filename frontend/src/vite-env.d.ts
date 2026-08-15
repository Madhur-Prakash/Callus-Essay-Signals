/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Overrides the API base path. Defaults to `/api/v1`, which the Vite dev
   *  server proxies to the FastAPI process. */
  readonly VITE_API_BASE_URL?: string;
  /** Dev-only: where the Vite proxy forwards `/api` requests. */
  readonly VITE_API_TARGET?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
