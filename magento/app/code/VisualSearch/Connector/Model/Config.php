<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model;

use Magento\Framework\App\Config\ScopeConfigInterface;
use Magento\Framework\Encryption\EncryptorInterface;

/**
 * Typed access to the `visual_search/*` system configuration.
 */
class Config
{
    private const PATH_ENABLED = 'visual_search/general/enabled';
    private const PATH_API_URL = 'visual_search/general/api_url';
    private const PATH_API_KEY = 'visual_search/general/api_key';
    private const PATH_TIMEOUT = 'visual_search/general/timeout';
    private const PATH_AUTO_SYNC = 'visual_search/sync/auto_sync';
    private const PATH_AUTO_SYNC_MODE = 'visual_search/sync/auto_sync_mode';
    private const PATH_CHUNK_SIZE = 'visual_search/sync/chunk_size';
    private const PATH_COLOR_ATTRIBUTE = 'visual_search/sync/color_attribute';
    private const PATH_LOG_RETENTION = 'visual_search/sync/log_retention_days';
    private const PATH_MATCH_CATEGORY = 'visual_search/search/match_category';
    private const PATH_MATCH_COLOR = 'visual_search/search/match_color';
    private const PATH_TRACK_EVENTS = 'visual_search/recommendations/track_events';
    private const PATH_RECOMMENDATIONS = 'visual_search/recommendations/enabled';
    private const PATH_RECOMMENDATION_LIMIT = 'visual_search/recommendations/limit';

    /** The API rejects import batches larger than this (MAX_BULK_IMPORT_ITEMS default). */
    public const MAX_CHUNK_SIZE = 100;

    public function __construct(
        private readonly ScopeConfigInterface $scopeConfig,
        private readonly EncryptorInterface $encryptor
    ) {
    }

    public function isEnabled(): bool
    {
        return $this->scopeConfig->isSetFlag(self::PATH_ENABLED);
    }

    public function getApiUrl(): string
    {
        return rtrim(trim((string)$this->scopeConfig->getValue(self::PATH_API_URL)), '/');
    }

    public function getApiKey(): string
    {
        $stored = (string)$this->scopeConfig->getValue(self::PATH_API_KEY);
        return $stored === '' ? '' : $this->encryptor->decrypt($stored);
    }

    public function getTimeout(): int
    {
        return max(1, (int)$this->scopeConfig->getValue(self::PATH_TIMEOUT) ?: 30);
    }

    public function isAutoSyncEnabled(): bool
    {
        return $this->isEnabled() && $this->scopeConfig->isSetFlag(self::PATH_AUTO_SYNC);
    }

    public function getAutoSyncMode(): string
    {
        return (string)$this->scopeConfig->getValue(self::PATH_AUTO_SYNC_MODE) ?: 'immediate';
    }

    public function getChunkSize(): int
    {
        $size = (int)$this->scopeConfig->getValue(self::PATH_CHUNK_SIZE) ?: 50;
        return min(self::MAX_CHUNK_SIZE, max(1, $size));
    }

    public function getColorAttribute(): string
    {
        return trim((string)$this->scopeConfig->getValue(self::PATH_COLOR_ATTRIBUTE));
    }

    public function getLogRetentionDays(): int
    {
        return max(1, (int)$this->scopeConfig->getValue(self::PATH_LOG_RETENTION) ?: 30);
    }

    public function shouldMatchCategory(): bool
    {
        return $this->scopeConfig->isSetFlag(self::PATH_MATCH_CATEGORY);
    }

    public function shouldMatchColor(): bool
    {
        return $this->scopeConfig->isSetFlag(self::PATH_MATCH_COLOR);
    }

    /** Whether shopper behavior (views, add-to-cart, purchases) is reported to the service. */
    public function isTrackingEnabled(): bool
    {
        return $this->isEnabled() && $this->scopeConfig->isSetFlag(self::PATH_TRACK_EVENTS);
    }

    public function areRecommendationsEnabled(): bool
    {
        return $this->isEnabled() && $this->scopeConfig->isSetFlag(self::PATH_RECOMMENDATIONS);
    }

    public function getRecommendationLimit(): int
    {
        return min(20, max(1, (int)$this->scopeConfig->getValue(self::PATH_RECOMMENDATION_LIMIT) ?: 6));
    }
}
