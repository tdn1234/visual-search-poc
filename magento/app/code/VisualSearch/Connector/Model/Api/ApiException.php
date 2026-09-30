<?php
declare(strict_types=1);

namespace VisualSearch\Connector\Model\Api;

/**
 * Transport-level failure (unreachable host, timeout, missing config).
 * An HTTP error *response* is not an exception -- see {@see Response}.
 */
class ApiException extends \RuntimeException
{
}
