"""
Test Mock Settings for Kingst Logic Analyzer

This module provides test-specific mock settings implementations that can be used
in unit tests without requiring the full C++ functionality.
"""

from kingst_analyzer.types import Channel, DisplayBase

class MockAnalyzerSettings:
    """
    Mock implementation of AnalyzerSettings for testing.
    
    This is a pure Python implementation that mimics the behavior of the C++ AnalyzerSettings class
    without actually inheriting from it. This avoids the issues with protected methods.
    """
    
    def __init__(self, name="Test Settings"):
        """Initialize a mock analyzer settings object."""
        self.name = name
        self._settings_str = "{}"
        self._error_text = ""
        # Store data in dictionaries for easy access
        self.channels = {}
        self.interfaces = []
        self.export_options = []
        self._use_system_display_base = False
        self._display_base = 0  # Decimal
    
    def get_name(self):
        """Get the name of these settings."""
        return self.name
    
    # Channel management
    def add_channel(self, channel, label, is_used=True):
        """
        Add a channel to the settings.
        
        Args:
            channel: Channel object
            label: Label for the channel
            is_used: Whether the channel is used
        """
        self.channels[label] = (channel, is_used)
    
    # For compatibility with existing code
    add_channel_wrapper = add_channel
    
    def clear_channels(self):
        """Clear all channels."""
        self.channels.clear()
    
    def get_channels_count(self):
        """Get the number of channels."""
        return len(self.channels)
    
    def get_channel(self, index):
        """
        Get a channel by index.
        
        Args:
            index: Channel index
            
        Returns:
            tuple: (channel, label, is_used)
        """
        if index >= len(self.channels):
            return None, "", False
            
        label = list(self.channels.keys())[index]
        channel, is_used = self.channels[label]
        return channel, label, is_used
    
    # Interface management
    def add_interface(self, interface):
        """
        Add an interface to the settings.
        
        Args:
            interface: Interface object
        """
        self.interfaces.append(interface)
    
    # For compatibility with existing code
    add_interface_wrapper = add_interface
    
    def get_settings_interfaces_count(self):
        """Get the number of interfaces."""
        return len(self.interfaces)
    
    def get_settings_interface(self, index):
        """
        Get an interface by index.
        
        Args:
            index: Interface index
            
        Returns:
            Interface object
        """
        if index >= len(self.interfaces):
            return None
        return self.interfaces[index]
    
    # Error handling
    def set_error_text(self, error_text):
        """
        Set error text for invalid settings.
        
        Args:
            error_text: Error message
        """
        self._error_text = error_text
    
    # For compatibility with existing code
    set_error_text_wrapper = set_error_text
    
    def get_save_error_message(self):
        """
        Get the error message.
        
        Returns:
            str: Error message
        """
        return self._error_text
    
    # Export options
    def add_export_option(self, user_id, menu_text):
        """
        Add an export option.
        
        Args:
            user_id: User ID for the option
            menu_text: Menu text
        """
        self.export_options.append((user_id, menu_text))
    
    def get_export_options_count(self):
        """Get the number of export options."""
        return len(self.export_options)
    
    def get_export_option(self, index):
        """
        Get an export option by index.
        
        Args:
            index: Option index
            
        Returns:
            tuple: (user_id, menu_text)
        """
        if index >= len(self.export_options):
            return 0, ""
        return self.export_options[index]
    
    # Display settings
    def get_use_system_display_base(self):
        """
        Check if using system display base.
        
        Returns:
            bool: True if using system display base
        """
        return self._use_system_display_base
    
    def set_use_system_display_base(self, use_system):
        """
        Set whether to use system display base.
        
        Args:
            use_system: Whether to use system display base
        """
        self._use_system_display_base = use_system
    
    def get_analyzer_display_base(self):
        """
        Get the display base.
        
        Returns:
            int: Display base
        """
        return self._display_base
    
    def set_analyzer_display_base(self, display_base):
        """
        Set the display base.
        
        Args:
            display_base: Display base
        """
        self._display_base = display_base
    
    # Virtual methods that would be implemented by derived classes
    def set_settings_from_interfaces(self):
        """
        Apply settings from interfaces.
        
        Returns:
            bool: True if successful
        """
        return True
    
    def load_settings(self, settings_str):
        """
        Load settings from a string.
        
        Args:
            settings_str: Settings string
        """
        self._settings_str = settings_str
    
    def save_settings(self):
        """
        Save settings to a string.
        
        Returns:
            str: Settings string
        """
        return self._settings_str
    
    def get_setting_brief(self):
        """
        Get a brief description of the settings.
        
        Returns:
            str: Brief description
        """
        return f"Mock Settings '{self.name}'"


class SimpleSettings(MockAnalyzerSettings):
    """Simple settings class for testing."""
    
    def __init__(self):
        super().__init__("Simple Test Settings")
        self.channel = Channel(0, 0)
        self.add_channel(self.channel, "Test Channel", True)


class SpiAnalyzerSettings(MockAnalyzerSettings):
    """Settings for the SPI Analyzer."""
    
    def __init__(self):
        super().__init__("SPI Analyzer Settings")
        
        # Default channels
        self.clock_channel = Channel(0, 0)  # Device 0, Channel 0
        self.mosi_channel = Channel(0, 1)   # Device 0, Channel 1
        self.miso_channel = Channel(0, 2)   # Device 0, Channel 2
        self.enable_channel = Channel(0, 3) # Device 0, Channel 3
        
        # Add channels to the settings
        self.add_channel(self.clock_channel, "Clock", True)
        self.add_channel(self.mosi_channel, "MOSI", True)
        self.add_channel(self.miso_channel, "MISO", True)
        self.add_channel(self.enable_channel, "Enable", True)