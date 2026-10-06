<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Controller\Chat;

use Magento\Framework\App\Action\HttpPostActionInterface;
use Magento\Framework\App\Request\Http as HttpRequest;
use Magento\Framework\Controller\Result\Json;
use Magento\Framework\Controller\Result\JsonFactory;
use Magento\Framework\Exception\LocalizedException;
use Psr\Log\LoggerInterface;
use VisualSearch\Connector\Model\Chat\AgentChat;
use VisualSearch\Connector\Model\Config;

/**
 * POST /visualsearch/chat/send -- one turn with the shopping assistant, as JSON.
 *
 * Fields: `message` (required), `session_id` (optional, echoed back by the
 * previous reply), `image` (optional upload). Magento's storefront CSRF check
 * applies automatically (the widget posts `form_key`). The upload is validated
 * by content (getimagesize), never by the client-supplied name or MIME type.
 * Responses are never cached.
 */
class Send implements HttpPostActionInterface
{
    private const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
    private const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
    private const SESSION_ID_PATTERN = '/^[A-Za-z0-9_-]{1,64}$/';

    public function __construct(
        private readonly HttpRequest $request,
        private readonly JsonFactory $jsonFactory,
        private readonly Config $config,
        private readonly AgentChat $agentChat,
        private readonly LoggerInterface $logger
    ) {
    }

    public function execute(): Json
    {
        $result = $this->jsonFactory->create();
        $result->setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0', true);

        if (!$this->config->isChatEnabled()) {
            return $result->setHttpResponseCode(404)->setData(['error' => (string)__('The assistant is not available.')]);
        }

        try {
            $message = trim((string)$this->request->getPostValue('message'));
            if ($message === '' || mb_strlen($message) > AgentChat::MAX_MESSAGE_LENGTH) {
                throw new LocalizedException(__('Please enter a message (up to %1 characters).', AgentChat::MAX_MESSAGE_LENGTH));
            }

            $sessionId = (string)$this->request->getPostValue('session_id');
            if (preg_match(self::SESSION_ID_PATTERN, $sessionId) !== 1) {
                $sessionId = null;
            }

            return $result->setData($this->agentChat->send($message, $sessionId, $this->readImage()));
        } catch (LocalizedException $e) {
            return $result->setHttpResponseCode(400)->setData(['error' => $e->getMessage()]);
        } catch (\Throwable $e) {
            $this->logger->error('Shopping assistant failed: ' . $e->getMessage(), ['exception' => $e]);
            return $result->setHttpResponseCode(500)
                ->setData(['error' => (string)__('Something went wrong. Please try again.')]);
        }
    }

    /**
     * @return array{bytes: string, mime: string, filename: string}|null null when no image was attached
     * @throws LocalizedException
     */
    private function readImage(): ?array
    {
        $file = $this->request->getFiles('image');
        if (!is_array($file) || ($file['error'] ?? UPLOAD_ERR_NO_FILE) === UPLOAD_ERR_NO_FILE) {
            return null;
        }

        $tmpName = (string)($file['tmp_name'] ?? '');
        if (($file['error'] ?? UPLOAD_ERR_NO_FILE) !== UPLOAD_ERR_OK || $tmpName === '' || !is_uploaded_file($tmpName)) {
            throw new LocalizedException(__('That image could not be uploaded. Please try again.'));
        }
        if ((int)$file['size'] > self::MAX_IMAGE_BYTES) {
            throw new LocalizedException(__('That image is too large. The limit is 5 MB.'));
        }

        $info = @getimagesize($tmpName);
        $mime = is_array($info) ? (string)$info['mime'] : '';
        if (!in_array($mime, self::ALLOWED_MIME_TYPES, true)) {
            throw new LocalizedException(__('Please upload a JPEG, PNG or WEBP image.'));
        }

        return [
            'bytes' => (string)file_get_contents($tmpName),
            'mime' => $mime,
            'filename' => 'chat.' . ['image/jpeg' => 'jpg', 'image/png' => 'png', 'image/webp' => 'webp'][$mime],
        ];
    }
}
