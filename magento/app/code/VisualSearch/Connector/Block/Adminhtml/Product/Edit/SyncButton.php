<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Block\Adminhtml\Product\Edit;

use Magento\Framework\AuthorizationInterface;
use Magento\Framework\App\RequestInterface;
use Magento\Framework\UrlInterface;
use Magento\Framework\View\Element\UiComponent\Control\ButtonProviderInterface;
use VisualSearch\Connector\Model\Config;

/**
 * "Sync to Visual Search" button in the product edit page's button bar.
 */
class SyncButton implements ButtonProviderInterface
{
    public function __construct(
        private readonly RequestInterface $request,
        private readonly UrlInterface $urlBuilder,
        private readonly AuthorizationInterface $authorization,
        private readonly Config $config
    ) {
    }

    public function getButtonData(): array
    {
        $productId = (int)$this->request->getParam('id');

        // Only for saved products, and only if the module is on and the admin may use it.
        if ($productId === 0
            || !$this->config->isEnabled()
            || !$this->authorization->isAllowed('VisualSearch_Connector::sync')
        ) {
            return [];
        }

        $url = $this->urlBuilder->getUrl('visualsearch/sync/single', ['id' => $productId]);

        return [
            'label' => __('Sync to Visual Search'),
            'class' => 'action-secondary',
            'on_click' => sprintf("location.href = '%s';", $url),
            'sort_order' => 25,
        ];
    }
}
