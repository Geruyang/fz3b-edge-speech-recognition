static bool fz_of_legacy_idle(void)
{
	static const char * const paths[] = {
		"/fpga-driver@0", "/amba_pl@0/reg_irq_path@a0000000",
		"/amba_pl@0/fz3b-cnn-ddr-v10-signed40@a4000000",
		"/amba_pl@0/mipi_axi@a0010000", "/amba_pl@0/serial@a0020000",
	};
	struct device_node *node;
	struct platform_device *pdev;
	unsigned int i;
	bool occupied;

	for (i = 0; i < ARRAY_SIZE(paths); ++i) {
		node = of_find_node_by_path(paths[i]);
		if (!node)
			continue;
		pdev = of_find_device_by_node(node);
		occupied = of_device_is_available(node);
		if (pdev) {
			device_lock(&pdev->dev);
			occupied |= !!pdev->dev.driver;
			device_unlock(&pdev->dev);
			put_device(&pdev->dev);
		}
		of_node_put(node);
		if (occupied)
			return false;
	}
	return true;
}
