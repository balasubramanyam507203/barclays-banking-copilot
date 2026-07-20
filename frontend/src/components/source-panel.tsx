"use client";

import type {
  SourceDocument,
  SourceReference,
} from "@/lib/types";

interface SourcePanelProps {
  source: SourceReference | null;
  document: SourceDocument | null;
  isLoading: boolean;
  error: string | null;
  onClose: () => void;
}

export default function SourcePanel({
  source,
  document,
  isLoading,
  error,
  onClose,
}: SourcePanelProps) {
  if (source === null) {
    return null;
  }

  return (
    <div
      className="sourceOverlay"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <aside
        className="sourcePanel"
        aria-label="Policy source details"
      >
        <header className="sourcePanelHeader">
          <div>
            <p className="sourceLabel">
              {source.label}
            </p>

            <h2>{source.title}</h2>

            <p className="sourceMetadata">
              {source.document_id} · Version{" "}
              {source.version}
            </p>
          </div>

          <button
            type="button"
            className="closeButton"
            onClick={onClose}
            aria-label="Close source panel"
          >
            ×
          </button>
        </header>

        <div className="sourcePanelBody">
          <section className="sourceSummary">
            <p>
              <strong>Citation</strong>
            </p>

            <p>{source.citation}</p>

            <p>
              <strong>Source location</strong>
            </p>

            <p className="sourcePath">
              {source.source}
            </p>
          </section>

          {isLoading && (
            <div
              className="sourceLoading"
              role="status"
            >
              <span className="spinner" />
              Loading authorized source…
            </div>
          )}

          {error !== null && (
            <div
              className="errorBanner"
              role="alert"
            >
              {error}
            </div>
          )}

          {!isLoading &&
            error === null &&
            document !== null && (
              <section className="sourceChunks">
                <div className="sourceDocumentHeading">
                  <h3>Authorized policy evidence</h3>

                  <span>
                    {document.chunks.length}{" "}
                    {document.chunks.length === 1
                      ? "chunk"
                      : "chunks"}
                  </span>
                </div>

                {document.chunks.map((chunk) => (
                  <article
                    className="sourceChunk"
                    key={chunk.chunk_id}
                  >
                    <div className="chunkHeader">
                      <span>
                        Chunk {chunk.chunk_number} of{" "}
                        {chunk.total_chunks}
                      </span>

                      <code>{chunk.chunk_id}</code>
                    </div>

                    <p>{chunk.content}</p>
                  </article>
                ))}
              </section>
            )}
        </div>
      </aside>
    </div>
  );
}