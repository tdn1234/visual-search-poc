<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Block;

use Magento\Framework\View\Element\Template;
use Magento\Framework\View\Element\Template\Context;
use VisualSearch\Connector\Model\Config;

/**
 * Placeholder for the shopping-assistant chat widget.
 *
 * Outputs only a container plus config for `js/chat.js`, which builds the UI
 * and talks to the module's JSON endpoint -- nothing personal is rendered
 * into the (full-page-cached) page.
 */
class Chat extends Template
{
    protected $_template = 'VisualSearch_Connector::chat.phtml';

    public function __construct(Context $context, private readonly Config $config, array $data = [])
    {
        parent::__construct($context, $data);
    }

    /** JSON for the container's `data-mage-init`. */
    public function getInitConfig(): string
    {
        return (string)json_encode([
            'VisualSearch_Connector/js/chat' => [
                'url' => $this->getUrl('visualsearch/chat/send'),
                'title' => (string)__('Shopping assistant'),
                'placeholder' => (string)__('Ask about our products...'),
                'greeting' => (string)__('Hi! Ask me about products, or attach a photo to find similar items.'),
            ],
        ]);
    }

    protected function _toHtml(): string
    {
        return $this->config->isChatEnabled() ? parent::_toHtml() : '';
    }
}
