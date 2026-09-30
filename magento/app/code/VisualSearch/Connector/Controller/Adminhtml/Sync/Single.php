<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Controller\Adminhtml\Sync;

use Magento\Backend\App\Action;
use Magento\Backend\App\Action\Context;
use Magento\Backend\Model\View\Result\Redirect;
use Magento\Framework\App\Action\HttpGetActionInterface;
use VisualSearch\Connector\Model\Config;
use VisualSearch\Connector\Model\Source\Status;
use VisualSearch\Connector\Model\Source\SyncType;
use VisualSearch\Connector\Model\SyncService;

/**
 * "Sync to Visual Search" button on the product edit page: syncs that one
 * product immediately and reports the result. (GET is fine here: admin URLs
 * carry a secret key, which is Magento's CSRF protection for admin links.)
 */
class Single extends Action implements HttpGetActionInterface
{
    public const ADMIN_RESOURCE = 'VisualSearch_Connector::sync';

    public function __construct(
        Context $context,
        private readonly Config $config,
        private readonly SyncService $syncService
    ) {
        parent::__construct($context);
    }

    public function execute(): Redirect
    {
        $productId = (int)$this->getRequest()->getParam('id');
        $redirect = $this->resultRedirectFactory->create();

        if ($productId === 0) {
            $this->messageManager->addErrorMessage(__('No product selected.'));
            return $redirect->setPath('catalog/product');
        }
        $redirect->setPath('catalog/product/edit', ['id' => $productId]);

        if (!$this->config->isEnabled()) {
            $this->messageManager->addErrorMessage(__('The Visual Search module is disabled.'));
            return $redirect;
        }

        $result = $this->syncService->syncNow($productId, SyncType::MANUAL);
        match ($result['status']) {
            Status::SUCCESS => $this->messageManager->addSuccessMessage(__($result['message'])),
            Status::SKIPPED => $this->messageManager->addNoticeMessage(__($result['message'])),
            default => $this->messageManager->addErrorMessage(__('Visual search sync failed: %1', $result['message'])),
        };

        return $redirect;
    }
}
