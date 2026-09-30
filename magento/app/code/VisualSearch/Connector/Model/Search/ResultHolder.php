<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Search;

use Magento\Catalog\Model\Product;

/**
 * Request-scoped hand-off between the search controller and the results
 * block (DI shares one instance per request). Simpler and more reliable than
 * reaching into the not-yet-generated layout from the controller.
 */
class ResultHolder
{
    /** @var array<int, array{product: Product, score: float}>|null null = no search was run */
    private ?array $results = null;
    private ?string $error = null;

    /**
     * @param array<int, array{product: Product, score: float}> $results
     */
    public function setResults(array $results): void
    {
        $this->results = $results;
    }

    public function setError(string $error): void
    {
        $this->error = $error;
    }

    /**
     * @return array<int, array{product: Product, score: float}>|null
     */
    public function getResults(): ?array
    {
        return $this->results;
    }

    public function getError(): ?string
    {
        return $this->error;
    }
}
