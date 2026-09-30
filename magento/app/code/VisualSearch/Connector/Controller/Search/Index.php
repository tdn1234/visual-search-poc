<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Controller\Search;

use Magento\Framework\App\Action\HttpGetActionInterface;
use Magento\Framework\Controller\ResultInterface;
use Magento\Framework\Controller\Result\ForwardFactory;
use Magento\Framework\View\Result\PageFactory;
use VisualSearch\Connector\Model\Config;

/**
 * GET /visualsearch/search -- the "search by image" upload page.
 */
class Index implements HttpGetActionInterface
{
    public function __construct(
        private readonly Config $config,
        private readonly PageFactory $pageFactory,
        private readonly ForwardFactory $forwardFactory
    ) {
    }

    public function execute(): ResultInterface
    {
        if (!$this->config->isEnabled()) {
            return $this->forwardFactory->create()->forward('noroute');
        }

        $page = $this->pageFactory->create();
        $page->getConfig()->getTitle()->set(__('Search by Image'));
        return $page;
    }
}
