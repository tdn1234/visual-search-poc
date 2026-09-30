<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Api;

/**
 * A decoded HTTP response from the visual search API.
 */
class Response
{
    /**
     * @param array<string, mixed> $data decoded JSON body ([] if not JSON)
     */
    public function __construct(
        private readonly int $statusCode,
        private readonly array $data,
        private readonly ?int $retryAfter = null
    ) {
    }

    public function getStatusCode(): int
    {
        return $this->statusCode;
    }

    public function isSuccess(): bool
    {
        return $this->statusCode >= 200 && $this->statusCode < 300;
    }

    /**
     * @return array<string, mixed>
     */
    public function getData(): array
    {
        return $this->data;
    }

    /** Seconds from the `Retry-After` header (sent with 429s), if any. */
    public function getRetryAfter(): ?int
    {
        return $this->retryAfter;
    }

    /** FastAPI puts the reason in `detail` (a string, or a list for validation errors). */
    public function getErrorMessage(): string
    {
        $detail = $this->data['detail'] ?? null;
        if (is_string($detail) && $detail !== '') {
            return $detail;
        }
        if (is_array($detail)) {
            return (string)json_encode($detail);
        }
        return 'HTTP ' . $this->statusCode;
    }
}
