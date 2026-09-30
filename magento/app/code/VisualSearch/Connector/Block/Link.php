<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Block;

use Magento\Framework\View\Element\Template;
use Magento\Framework\View\Element\Template\Context;
use VisualSearch\Connector\Model\Config;

/**
 * "Search by image" header link.
 */
class Link extends Template
{
    protected $_template = 'VisualSearch_Connector::link.phtml';

    public function __construct(
        Context $context,
        private readonly Config $config,
        array $data = []
    ) {
        parent::__construct($context, $data);
    }

    public function getSearchUrl(): string
    {
        return $this->getUrl('visualsearch/search');
    }

    protected function _toHtml(): string
    {
        return $this->config->isEnabled() ? parent::_toHtml() : '';
    }
}
