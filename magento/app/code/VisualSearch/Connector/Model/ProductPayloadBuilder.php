<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model;

use Magento\Catalog\Model\Product;
use Magento\Framework\App\Filesystem\DirectoryList;
use Magento\Framework\Exception\LocalizedException;
use Magento\Framework\Filesystem;

/**
 * Turns a Magento product into what the visual search API accepts.
 *
 * The API is stricter than Magento, so this is where a product is either
 * made API-shaped or rejected with a reason the admin can act on (which
 * ends up in the sync log):
 *  - sku: lowercase letters/digits/hyphens only (API SKU_PATTERN)
 *  - price: must be > 0
 *  - image: JPEG or PNG only, and the product needs one
 */
class ProductPayloadBuilder
{
    private const ALLOWED_MIME_TO_FILENAME = [
        'image/jpeg' => 'image.jpg',
        'image/png' => 'image.png',
    ];

    public function __construct(
        private readonly Config $config,
        private readonly Filesystem $filesystem
    ) {
    }

    /**
     * Normalize a Magento SKU into the API's slug format, e.g. "Shoe_Red 42" -> "shoe-red-42".
     * Deterministic, so the same product always maps to the same API sku.
     * Falls back to "product-<id>" if nothing usable is left (e.g. an all-symbol SKU).
     */
    public static function toApiSku(string $sku, int $productId): string
    {
        $slug = trim((string)preg_replace('/[^a-z0-9]+/', '-', strtolower($sku)), '-');
        return $slug !== '' ? substr($slug, 0, 64) : 'product-' . $productId;
    }

    /**
     * @return array{
     *     magento_sku: string,
     *     meta: array{sku: string, name: string, price: string, category: string, color: ?string},
     *     image: string,
     *     mime: string,
     *     filename: string
     * }
     * @throws LocalizedException if the product can't be represented in the API
     */
    public function build(Product $product): array
    {
        $productId = (int)$product->getId();
        $magentoSku = (string)$product->getSku();

        $price = $this->resolvePrice($product);
        [$image, $mime] = $this->readImage($product);

        return [
            'magento_sku' => $magentoSku,
            'meta' => [
                'sku' => self::toApiSku($magentoSku, $productId),
                'name' => (string)$product->getName(),
                'price' => number_format($price, 2, '.', ''),
                'category' => $this->resolveCategory($product),
                'color' => $this->resolveColor($product),
            ],
            'image' => $image,
            'mime' => $mime,
            'filename' => self::ALLOWED_MIME_TO_FILENAME[$mime],
        ];
    }

    private function resolvePrice(Product $product): float
    {
        // final_price's amount also works for configurable/bundle products,
        // whose own `price` attribute is empty.
        $price = (float)$product->getPriceInfo()->getPrice('final_price')->getAmount()->getValue();
        if ($price <= 0) {
            $price = (float)$product->getPrice();
        }
        if ($price <= 0) {
            throw new LocalizedException(__('Product has no price above 0 (the visual search API requires one).'));
        }
        return $price;
    }

    /** Deepest assigned category wins (most specific), e.g. "Shoes" over "Men". */
    private function resolveCategory(Product $product): string
    {
        $deepestName = '';
        $deepestLevel = -1;
        $categories = $product->getCategoryCollection()->addAttributeToSelect('name');
        foreach ($categories as $category) {
            $level = (int)$category->getLevel();
            if ($level > $deepestLevel && (string)$category->getName() !== '') {
                $deepestLevel = $level;
                $deepestName = (string)$category->getName();
            }
        }
        return $deepestName !== '' ? $deepestName : 'Uncategorized';
    }

    private function resolveColor(Product $product): ?string
    {
        $code = $this->config->getColorAttribute();
        if ($code === '') {
            return null;
        }
        // getAttributeText() fatals (getSource() on false) when the configured
        // attribute code doesn't exist in this store, so treat that as "no color".
        if (!$product->getResource()->getAttribute($code)) {
            return null;
        }
        $value = $product->getAttributeText($code);
        if (is_array($value)) {
            $value = reset($value);
        }
        $color = is_string($value) ? strtolower(trim($value)) : '';
        return $color !== '' ? $color : null;
    }

    /**
     * @return array{0: string, 1: string} [bytes, mime]
     * @throws LocalizedException
     */
    private function readImage(Product $product): array
    {
        $media = $this->filesystem->getDirectoryRead(DirectoryList::MEDIA);

        foreach (['image', 'small_image', 'thumbnail'] as $attribute) {
            $file = (string)$product->getData($attribute);
            if ($file === '' || $file === 'no_selection') {
                continue;
            }
            $relativePath = 'catalog/product/' . ltrim($file, '/');
            if (!$media->isFile($relativePath)) {
                continue;
            }

            $bytes = $media->readFile($relativePath);
            $info = @getimagesizefromstring($bytes);
            $mime = is_array($info) ? (string)$info['mime'] : '';
            if (!isset(self::ALLOWED_MIME_TO_FILENAME[$mime])) {
                throw new LocalizedException(__(
                    'Product image "%1" is %2; the visual search API accepts only JPEG or PNG.',
                    $file,
                    $mime !== '' ? $mime : 'not a readable image'
                ));
            }
            return [$bytes, $mime];
        }

        throw new LocalizedException(__('Product has no base image file on disk.'));
    }
}
