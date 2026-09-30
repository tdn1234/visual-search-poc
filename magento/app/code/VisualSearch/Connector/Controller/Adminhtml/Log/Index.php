<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Controller\Adminhtml\Log;

use Magento\Backend\App\Action;
use Magento\Backend\App\Action\Context;
use Magento\Framework\App\Action\HttpGetActionInterface;
use Magento\Framework\View\Result\Page;
use Magento\Framework\View\Result\PageFactory;

/**
 * Admin > Catalog > Visual Search Sync Log
 */
class Index extends Action implements HttpGetActionInterface
{
    public const ADMIN_RESOURCE = 'VisualSearch_Connector::logs';

    public function __construct(
        Context $context,
        private readonly PageFactory $resultPageFactory
    ) {
        parent::__construct($context);
    }

    public function execute(): Page
    {
        $page = $this->resultPageFactory->create();
        $page->setActiveMenu('VisualSearch_Connector::logs');
        $page->getConfig()->getTitle()->prepend(__('Visual Search Sync Log'));
        return $page;
    }
}
