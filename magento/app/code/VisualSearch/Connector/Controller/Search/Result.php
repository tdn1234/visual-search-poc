<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Controller\Search;

use Magento\Framework\App\Action\HttpPostActionInterface;
use Magento\Framework\App\Request\Http as HttpRequest;
use Magento\Framework\Controller\Result\ForwardFactory;
use Magento\Framework\Controller\ResultInterface;
use Magento\Framework\Exception\LocalizedException;
use Magento\Framework\View\Result\PageFactory;
use Psr\Log\LoggerInterface;
use VisualSearch\Connector\Model\Config;
use VisualSearch\Connector\Model\Search\ImageSearch;
use VisualSearch\Connector\Model\Search\ResultHolder;

/**
 * POST /visualsearch/search/result -- validates the uploaded image, runs the
 * search, and renders the results page.
 *
 * Magento's storefront CSRF check applies automatically (the form posts
 * `form_key`). The upload is validated by content (getimagesize), never by
 * the client-supplied filename or MIME type.
 */
class Result implements HttpPostActionInterface
{
    private const MAX_BYTES = 5 * 1024 * 1024;
    private const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

    public function __construct(
        private readonly HttpRequest $request,
        private readonly Config $config,
        private readonly PageFactory $pageFactory,
        private readonly ForwardFactory $forwardFactory,
        private readonly ImageSearch $imageSearch,
        private readonly ResultHolder $resultHolder,
        private readonly LoggerInterface $logger
    ) {
    }

    public function execute(): ResultInterface
    {
        if (!$this->config->isEnabled()) {
            return $this->forwardFactory->create()->forward('noroute');
        }

        try {
            [$bytes, $mime] = $this->readUpload();
            $this->resultHolder->setResults($this->imageSearch->search($bytes, $mime, $this->filenameFor($mime)));
        } catch (LocalizedException $e) {
            $this->resultHolder->setError($e->getMessage());
        } catch (\Throwable $e) {
            $this->logger->error('Visual search failed: ' . $e->getMessage(), ['exception' => $e]);
            $this->resultHolder->setError((string)__('Something went wrong while searching. Please try again.'));
        }

        $page = $this->pageFactory->create();
        $page->getConfig()->getTitle()->set(__('Search by Image'));
        return $page;
    }

    /**
     * @return array{0: string, 1: string} [bytes, mime]
     * @throws LocalizedException
     */
    private function readUpload(): array
    {
        $file = $this->request->getFiles('image');
        $tmpName = is_array($file) ? (string)($file['tmp_name'] ?? '') : '';

        if (!is_array($file) || ($file['error'] ?? UPLOAD_ERR_NO_FILE) !== UPLOAD_ERR_OK
            || $tmpName === '' || !is_uploaded_file($tmpName)
        ) {
            throw new LocalizedException(__('Please choose an image to search with.'));
        }
        if ((int)$file['size'] > self::MAX_BYTES) {
            throw new LocalizedException(__('That image is too large. The limit is 5 MB.'));
        }

        $info = @getimagesize($tmpName);
        $mime = is_array($info) ? (string)$info['mime'] : '';
        if (!in_array($mime, self::ALLOWED_MIME_TYPES, true)) {
            throw new LocalizedException(__('Please upload a JPEG, PNG or WEBP image.'));
        }

        return [(string)file_get_contents($tmpName), $mime];
    }

    private function filenameFor(string $mime): string
    {
        return 'query.' . ['image/jpeg' => 'jpg', 'image/png' => 'png', 'image/webp' => 'webp'][$mime];
    }
}
