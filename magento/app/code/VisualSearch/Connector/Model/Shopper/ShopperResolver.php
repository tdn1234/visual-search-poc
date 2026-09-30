<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Shopper;

use Magento\Customer\Model\Session as CustomerSession;
use Magento\Framework\App\Request\Http as HttpRequest;
use Magento\Framework\Stdlib\Cookie\CookieMetadataFactory;
use Magento\Framework\Stdlib\CookieManagerInterface;

/**
 * Who is this shopper, as far as the visual search service is concerned?
 *
 * The service only needs an *opaque* id that groups one person's events:
 *  - logged-in customer -> "c<customer id>"
 *  - guest              -> "g<24 random hex>", kept in a first-party cookie
 *
 * No name, email or address ever leaves Magento. Both forms match the
 * API's shopper-id pattern (letters, digits, "-", "_", up to 64 chars).
 * A guest who logs in gets a new (customer) id, so their guest history
 * doesn't follow them -- they start from the popular-items fallback again.
 */
class ShopperResolver
{
    public const COOKIE_NAME = 'vs_shopper';
    private const COOKIE_LIFETIME = 31536000; // 1 year
    private const GUEST_ID_PATTERN = '/^g[a-f0-9]{24}$/';

    /** Resolved once per request: setting a cookie doesn't make it readable until the *next* request. */
    private ?string $resolved = null;

    public function __construct(
        private readonly CustomerSession $customerSession,
        private readonly CookieManagerInterface $cookieManager,
        private readonly CookieMetadataFactory $cookieMetadataFactory,
        private readonly HttpRequest $request
    ) {
    }

    public function resolve(): string
    {
        if ($this->resolved !== null) {
            return $this->resolved;
        }

        if ($this->customerSession->isLoggedIn()) {
            return $this->resolved = 'c' . (int)$this->customerSession->getCustomerId();
        }

        $existing = (string)$this->cookieManager->getCookie(self::COOKIE_NAME);
        if (preg_match(self::GUEST_ID_PATTERN, $existing) === 1) {
            return $this->resolved = $existing;
        }

        $guestId = 'g' . bin2hex(random_bytes(12));
        try {
            $this->cookieManager->setPublicCookie(
                self::COOKIE_NAME,
                $guestId,
                $this->cookieMetadataFactory->createPublicCookieMetadata()
                    ->setDuration(self::COOKIE_LIFETIME)
                    ->setPath('/')
                    ->setHttpOnly(true)
                    ->setSecure($this->request->isSecure())
                    ->setSameSite('Lax')
            );
        } catch (\Exception) {
            // Can't persist (headers already sent, cookie too large...): this request
            // still gets a usable id, it just won't be recognized on the next one.
        }
        return $this->resolved = $guestId;
    }
}
