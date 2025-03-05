"""
Kingst Logic Analyzer Settings Module

This module provides a Pythonic interface to the Kingst Logic Analyzer settings system.
It wraps the C++ settings classes with intuitive Python classes that handle validation,
provide helpful error messages, and offer convenient methods for common configurations.

Key Components:
- Setting interfaces (channel, number list, integer, text, boolean)
- Base analyzer settings class for extending with custom protocols
- Settings builder for fluent configuration
- Helper functions for common protocol settings

Example:
    # Create simple protocol analyzer settings
    settings = SimpleSerialSettings()
    settings.data_channel.channel = Channel(0, 0)
    settings.clock_channel.channel = Channel(0, 1)
    settings.baud_rate.value = 9600
    
    # Using builder pattern
    settings = (SettingsBuilder()
                .add_channel("data", "Data Line")
                .add_channel("clock", "Clock Line")
                .add_number_list("baud", "Baud Rate", options=[9600, 115200])
                .build())
"""

from typing import List, Dict, Union, Optional, Tuple, Any, Callable, TypeVar, Generic, cast
import json
import abc
from enum import Enum

# Import the C++ bound classes
from ._core import (
    AnalyzerSettings as CppAnalyzerSettings,
    AnalyzerSettingInterface,
    AnalyzerSettingInterfaceChannel,
    AnalyzerSettingInterfaceNumberList,
    AnalyzerSettingInterfaceInteger,
    AnalyzerSettingInterfaceText,
    AnalyzerSettingInterfaceBool,
    AnalyzerInterfaceTypeId,
    Channel,
    DisplayBase
)

__all__ = [
    'AnalyzerSettings',
    'SettingInterface',
    'ChannelSetting',
    'NumberListSetting',
    'IntegerSetting',
    'TextSetting',
    'BoolSetting',
    'SimpleAnalyzerSettings',
    'SettingsBuilder',
    'TextFieldType',
    'create_uart_settings',
    'create_spi_settings',
    'create_i2c_settings',
    'create_common_baud_rates',
    'create_common_bit_counts',
]

# Type variable for generic settings
T = TypeVar('T')


class TextFieldType(Enum):
    """Types of text input fields."""
    TEXT = AnalyzerSettingInterfaceText.TextType.NormalText
    FILE_PATH = AnalyzerSettingInterfaceText.TextType.FilePath
    FOLDER_PATH = AnalyzerSettingInterfaceText.TextType.FolderPath


class SettingInterface(abc.ABC):
    """
    Base class for all settings interfaces.
    
    This abstract class defines the common properties and methods that all
    setting interfaces should implement, providing a consistent API.
    """
    
    def __init__(self, title: str = "", tooltip: str = ""):
        """
        Initialize a setting interface.
        
        Args:
            title: Title/label for the setting
            tooltip: Tooltip text explaining the setting
        """
        self._cpp_interface: Optional[AnalyzerSettingInterface] = None
        self._title = title
        self._tooltip = tooltip
    
    @property
    def cpp_interface(self) -> AnalyzerSettingInterface:
        """Get the underlying C++ interface object."""
        if self._cpp_interface is None:
            raise RuntimeError("C++ interface not initialized")
        return self._cpp_interface
    
    @property
    def title(self) -> str:
        """Get the title/label for this setting."""
        if self._cpp_interface is None:
            return self._title
        return self._cpp_interface.get_title()
    
    @title.setter
    def title(self, value: str) -> None:
        """Set the title/label for this setting."""
        self._title = value
        if self._cpp_interface is not None:
            self._cpp_interface.set_title_and_tooltip(value, self.tooltip)
    
    @property
    def tooltip(self) -> str:
        """Get the tooltip text for this setting."""
        if self._cpp_interface is None:
            return self._tooltip
        return self._cpp_interface.get_tool_tip()
    
    @tooltip.setter
    def tooltip(self, value: str) -> None:
        """Set the tooltip text for this setting."""
        self._tooltip = value
        if self._cpp_interface is not None:
            self._cpp_interface.set_title_and_tooltip(self.title, value)
    
    @property
    def disabled(self) -> bool:
        """Check if this setting is disabled."""
        if self._cpp_interface is None:
            return False
        return self._cpp_interface.is_disabled()
    
    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the setting to a dictionary for serialization."""
        return {
            "type": self.__class__.__name__,
            "title": self.title,
            "tooltip": self.tooltip,
        }
    
    @classmethod
    @abc.abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SettingInterface':
        """Create a setting from a dictionary."""
        raise NotImplementedError


class ChannelSetting(SettingInterface):
    """
    Setting for selecting a channel.
    
    This interface allows the user to select a channel from the available channels.
    
    Example:
        ```python
        # Create a channel setting for selecting the clock input
        clk_setting = ChannelSetting(
            title="Clock Channel",
            tooltip="Select the channel connected to the clock signal",
            channel=Channel(0, 0),  # Initial selection: device 0, channel 0
            allow_none=False  # Require a channel selection (no "None" option)
        )
        
        # Access the selected channel
        clock_channel = clk_setting.channel
        
        # Change the selection
        clk_setting.channel = Channel(0, 1)  # Switch to device 0, channel 1
        ```
    """
    
    def __init__(
        self, 
        title: str = "", 
        tooltip: str = "", 
        channel: Optional[Channel] = None, 
        allow_none: bool = False
    ):
        """
        Initialize a channel selection setting.
        
        Args:
            title: Title/label for the setting
            tooltip: Tooltip text explaining the setting
            channel: Initial channel selection (optional)
            allow_none: Whether "None" is a valid selection
        """
        super().__init__(title, tooltip)
        self._cpp_interface = AnalyzerSettingInterfaceChannel()
        self._cpp_interface.set_title_and_tooltip(title, tooltip)
        self._cpp_interface.set_selection_of_none_is_allowed(allow_none)
        
        if channel is not None:
            self._cpp_interface.set_channel(channel)
    
    @property
    def channel(self) -> Channel:
        """Get the currently selected channel."""
        return self.cpp_interface.get_channel()
    
    @channel.setter
    def channel(self, value: Channel) -> None:
        """Set the selected channel."""
        self.cpp_interface.set_channel(value)
    
    @property
    def allow_none(self) -> bool:
        """Check if selecting "None" is allowed for this channel setting."""
        return cast(AnalyzerSettingInterfaceChannel, self.cpp_interface).get_selection_of_none_is_allowed()
    
    @allow_none.setter
    def allow_none(self, value: bool) -> None:
        """Set whether selecting "None" is allowed for this channel setting."""
        cast(AnalyzerSettingInterfaceChannel, self.cpp_interface).set_selection_of_none_is_allowed(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the setting to a dictionary for serialization."""
        data = super().to_dict()
        channel = self.channel
        data.update({
            "allow_none": self.allow_none,
            "channel": {
                "device_index": channel.device_index,
                "channel_index": channel.channel_index
            } if channel else None
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChannelSetting':
        """Create a channel setting from a dictionary."""
        channel_data = data.get("channel")
        channel = None
        if channel_data:
            channel = Channel(
                channel_data["device_index"],
                channel_data["channel_index"]
            )
        
        return cls(
            title=data.get("title", ""),
            tooltip=data.get("tooltip", ""),
            channel=channel,
            allow_none=data.get("allow_none", False)
        )


class NumberListSetting(SettingInterface):
    """
    Setting for selecting from a list of numeric values.
    
    This interface is used for settings where the user selects from a list of options,
    each with an associated numeric value (e.g., baud rates, bit counts, etc.).
    
    Example:
        ```python
        # Create a baud rate setting
        baud_setting = NumberListSetting(
            title="Baud Rate",
            tooltip="Select the communication baud rate"
        )
        
        # Add options
        baud_setting.add_option(9600, "9600 bps", "Standard serial rate")
        baud_setting.add_option(115200, "115.2 kbps", "High speed serial rate")
        
        # Set the selected value
        baud_setting.value = 9600
        
        # Access the selected value
        rate = baud_setting.value
        ```
    """
    
    def __init__(
        self, 
        title: str = "", 
        tooltip: str = "", 
        options: Optional[List[Union[float, Tuple[float, str, str]]]] = None
    ):
        """
        Initialize a number list selection setting.
        
        Args:
            title: Title/label for the setting
            tooltip: Tooltip text explaining the setting
            options: List of options to add. Can be:
                     - List of values
                     - List of (value, display_text, tooltip) tuples
        """
        super().__init__(title, tooltip)
        self._cpp_interface = AnalyzerSettingInterfaceNumberList()
        self._cpp_interface.set_title_and_tooltip(title, tooltip)
        
        if options:
            for option in options:
                if isinstance(option, tuple):
                    value, text, option_tooltip = option
                    self.add_option(value, text, option_tooltip)
                else:
                    self.add_option(option, str(option), "")
    
    @property
    def value(self) -> float:
        """Get the currently selected value."""
        return cast(AnalyzerSettingInterfaceNumberList, self.cpp_interface).get_number()
    
    @value.setter
    def value(self, number: float) -> None:
        """Set the selected value."""
        cast(AnalyzerSettingInterfaceNumberList, self.cpp_interface).set_number(number)
    
    def add_option(self, value: float, display_text: str, tooltip: str = "") -> None:
        """
        Add an option to the list.
        
        Args:
            value: Numeric value
            display_text: Display text for the option
            tooltip: Tooltip for the option
        """
        cast(AnalyzerSettingInterfaceNumberList, self.cpp_interface).add_number(value, display_text, tooltip)
    
    def clear_options(self) -> None:
        """Clear all options from the list."""
        cast(AnalyzerSettingInterfaceNumberList, self.cpp_interface).clear_numbers()
    
    def get_options(self) -> List[Tuple[float, str, str]]:
        """
        Get all options in the list.
        
        Returns:
            List of (value, display_text, tooltip) tuples for all options
        """
        interface = cast(AnalyzerSettingInterfaceNumberList, self.cpp_interface)
        count = interface.get_listbox_numbers_count()
        options = []
        
        for i in range(count):
            value = interface.get_listbox_number(i)
            text = interface.get_listbox_string(i)
            tooltip = ""
            if i < interface.get_listbox_tooltips_count():
                tooltip = interface.get_listbox_tooltip(i)
            options.append((value, text, tooltip))
        
        return options
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the setting to a dictionary for serialization."""
        data = super().to_dict()
        data.update({
            "options": self.get_options(),
            "value": self.value
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NumberListSetting':
        """Create a number list setting from a dictionary."""
        setting = cls(
            title=data.get("title", ""),
            tooltip=data.get("tooltip", ""),
            options=data.get("options", [])
        )
        
        if "value" in data:
            setting.value = data["value"]
            
        return setting


class IntegerSetting(SettingInterface):
    """
    Setting for entering an integer value.
    
    This interface allows the user to enter an integer value within a specified range.
    
    Example:
        ```python
        # Create a packet size setting
        size_setting = IntegerSetting(
            title="Packet Size",
            tooltip="Specify the packet size in bytes",
            value=64,  # Default value
            min_value=1,
            max_value=1024
        )
        
        # Access the current value
        packet_size = size_setting.value
        
        # Change the value
        size_setting.value = 128
        ```
    """
    
    def __init__(
        self, 
        title: str = "", 
        tooltip: str = "", 
        value: int = 0, 
        min_value: int = 0, 
        max_value: int = 100
    ):
        """
        Initialize an integer input setting.
        
        Args:
            title: Title/label for the setting
            tooltip: Tooltip text explaining the setting
            value: Initial value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
        """
        super().__init__(title, tooltip)
        self._cpp_interface = AnalyzerSettingInterfaceInteger()
        self._cpp_interface.set_title_and_tooltip(title, tooltip)
        self._cpp_interface.set_min(min_value)
        self._cpp_interface.set_max(max_value)
        self._cpp_interface.set_integer(value)
    
    @property
    def value(self) -> int:
        """Get the current integer value."""
        return cast(AnalyzerSettingInterfaceInteger, self.cpp_interface).get_integer()
    
    @value.setter
    def value(self, integer: int) -> None:
        """
        Set the integer value.
        
        Raises:
            ValueError: If the value is outside the allowed range
        """
        interface = cast(AnalyzerSettingInterfaceInteger, self.cpp_interface)
        min_val = interface.get_min()
        max_val = interface.get_max()
        
        if integer < min_val or integer > max_val:
            raise ValueError(f"Value must be between {min_val} and {max_val}")
            
        interface.set_integer(integer)
    
    @property
    def min_value(self) -> int:
        """Get the minimum allowed value."""
        return cast(AnalyzerSettingInterfaceInteger, self.cpp_interface).get_min()
    
    @min_value.setter
    def min_value(self, value: int) -> None:
        """Set the minimum allowed value."""
        cast(AnalyzerSettingInterfaceInteger, self.cpp_interface).set_min(value)
    
    @property
    def max_value(self) -> int:
        """Get the maximum allowed value."""
        return cast(AnalyzerSettingInterfaceInteger, self.cpp_interface).get_max()
    
    @max_value.setter
    def max_value(self, value: int) -> None:
        """Set the maximum allowed value."""
        cast(AnalyzerSettingInterfaceInteger, self.cpp_interface).set_max(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the setting to a dictionary for serialization."""
        data = super().to_dict()
        data.update({
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntegerSetting':
        """Create an integer setting from a dictionary."""
        return cls(
            title=data.get("title", ""),
            tooltip=data.get("tooltip", ""),
            value=data.get("value", 0),
            min_value=data.get("min_value", 0),
            max_value=data.get("max_value", 100)
        )


class TextSetting(SettingInterface):
    """
    Setting for entering text.
    
    This interface allows the user to enter text, which can be used for various purposes
    such as naming, entering file paths, etc.
    
    Example:
        ```python
        # Create a file path setting
        path_setting = TextSetting(
            title="Export Path",
            tooltip="Path where exported data will be saved",
            value="C:/exports",
            field_type=TextFieldType.FOLDER_PATH
        )
        
        # Access the text value
        export_path = path_setting.value
        
        # Change the value
        path_setting.value = "D:/new_exports"
        ```
    """
    
    def __init__(
        self, 
        title: str = "", 
        tooltip: str = "", 
        value: str = "", 
        field_type: TextFieldType = TextFieldType.TEXT
    ):
        """
        Initialize a text input setting.
        
        Args:
            title: Title/label for the setting
            tooltip: Tooltip text explaining the setting
            value: Initial text value
            field_type: Type of text field (normal text, file path, or folder path)
        """
        super().__init__(title, tooltip)
        self._cpp_interface = AnalyzerSettingInterfaceText()
        self._cpp_interface.set_title_and_tooltip(title, tooltip)
        self._cpp_interface.set_text(value)
        self._cpp_interface.set_text_type(field_type.value)
    
    @property
    def value(self) -> str:
        """Get the current text value."""
        return cast(AnalyzerSettingInterfaceText, self.cpp_interface).get_text()
    
    @value.setter
    def value(self, text: str) -> None:
        """Set the text value."""
        cast(AnalyzerSettingInterfaceText, self.cpp_interface).set_text(text)
    
    @property
    def field_type(self) -> TextFieldType:
        """Get the type of text field."""
        cpp_type = cast(AnalyzerSettingInterfaceText, self.cpp_interface).get_text_type()
        return TextFieldType(cpp_type)
    
    @field_type.setter
    def field_type(self, field_type: TextFieldType) -> None:
        """Set the type of text field."""
        cast(AnalyzerSettingInterfaceText, self.cpp_interface).set_text_type(field_type.value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the setting to a dictionary for serialization."""
        data = super().to_dict()
        data.update({
            "value": self.value,
            "field_type": self.field_type.name
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextSetting':
        """Create a text setting from a dictionary."""
        field_type = TextFieldType.TEXT
        if "field_type" in data:
            field_type = TextFieldType[data["field_type"]]
            
        return cls(
            title=data.get("title", ""),
            tooltip=data.get("tooltip", ""),
            value=data.get("value", ""),
            field_type=field_type
        )


class BoolSetting(SettingInterface):
    """
    Setting for boolean (checkbox) values.
    
    This interface presents a checkbox with a label for toggling boolean options.
    
    Example:
        ```python
        # Create an invert signal setting
        invert_setting = BoolSetting(
            title="Invert Signal",
            tooltip="Invert the signal polarity",
            value=False,
            checkbox_text="Invert"
        )
        
        # Access the boolean value
        is_inverted = invert_setting.value
        
        # Change the value
        invert_setting.value = True
        ```
    """
    
    def __init__(
        self, 
        title: str = "", 
        tooltip: str = "", 
        value: bool = False, 
        checkbox_text: str = ""
    ):
        """
        Initialize a boolean (checkbox) setting.
        
        Args:
            title: Title/label for the setting
            tooltip: Tooltip text explaining the setting
            value: Initial boolean value
            checkbox_text: Text to display next to the checkbox
        """
        super().__init__(title, tooltip)
        self._cpp_interface = AnalyzerSettingInterfaceBool()
        self._cpp_interface.set_title_and_tooltip(title, tooltip)
        self._cpp_interface.set_value(value)
        
        if checkbox_text:
            self._cpp_interface.set_check_box_text(checkbox_text)
    
    @property
    def value(self) -> bool:
        """Get the current boolean value."""
        return cast(AnalyzerSettingInterfaceBool, self.cpp_interface).get_value()
    
    @value.setter
    def value(self, value: bool) -> None:
        """Set the boolean value."""
        cast(AnalyzerSettingInterfaceBool, self.cpp_interface).set_value(value)
    
    @property
    def checkbox_text(self) -> str:
        """Get the text displayed next to the checkbox."""
        return cast(AnalyzerSettingInterfaceBool, self.cpp_interface).get_check_box_text()
    
    @checkbox_text.setter
    def checkbox_text(self, text: str) -> None:
        """Set the text displayed next to the checkbox."""
        cast(AnalyzerSettingInterfaceBool, self.cpp_interface).set_check_box_text(text)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the setting to a dictionary for serialization."""
        data = super().to_dict()
        data.update({
            "value": self.value,
            "checkbox_text": self.checkbox_text
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoolSetting':
        """Create a boolean setting from a dictionary."""
        return cls(
            title=data.get("title", ""),
            tooltip=data.get("tooltip", ""),
            value=data.get("value", False),
            checkbox_text=data.get("checkbox_text", "")
        )


class AnalyzerSettings(abc.ABC):
    """
    Base class for analyzer settings.
    
    This abstract class provides a Pythonic interface to the C++ AnalyzerSettings class.
    It manages the settings interfaces, load/save functionality, and validation.
    
    Must be subclassed to implement specific analyzer settings.
    
    Example:
        ```python
        class MyProtocolSettings(AnalyzerSettings):
            def __init__(self):
                super().__init__()
                
                # Create settings interfaces
                self.data_channel = ChannelSetting("Data Channel", "Select the data signal")
                self.clock_channel = ChannelSetting("Clock Channel", "Select the clock signal")
                self.baud_rate = NumberListSetting("Baud Rate", "Select the communication rate")
                
                # Add common baud rates
                common_rates = [(9600, "9600 bps", "Standard rate"),
                                (115200, "115.2 kbps", "High speed rate")]
                for rate, text, tooltip in common_rates:
                    self.baud_rate.add_option(rate, text, tooltip)
                
                # Register settings
                self.add_interface(self.data_channel)
                self.add_interface(self.clock_channel)
                self.add_interface(self.baud_rate)
                
            def from_json(self, json_str):
                # Custom JSON deserialization here
                pass
                
            def to_json(self):
                # Custom JSON serialization here
                pass
        ```
    """
    
    def __init__(self):
        """Initialize an analyzer settings object."""
        self._cpp_settings = self._create_cpp_settings()
        self._interfaces: List[SettingInterface] = []
        self._channels: Dict[str, Tuple[Channel, str, bool]] = {}
    
    @property
    def cpp_settings(self) -> CppAnalyzerSettings:
        """Get the underlying C++ settings object."""
        return self._cpp_settings
    
    @abc.abstractmethod
    def _create_cpp_settings(self) -> CppAnalyzerSettings:
        """
        Create the C++ settings implementation.
        
        Subclasses should implement this method to create a specific C++ settings object
        that implements the required virtual methods.
        
        Returns:
            The C++ settings implementation
        """
        raise NotImplementedError
    
    def add_interface(self, interface: SettingInterface) -> None:
        """
        Add a setting interface.
        
        Args:
            interface: Setting interface to add
        """
        self._interfaces.append(interface)
        self._cpp_settings.add_interface(interface.cpp_interface)
    
    def clear_channels(self) -> None:
        """Clear all reported channels."""
        self._cpp_settings.clear_channels()
        self._channels.clear()
    
    def add_channel(self, channel: Channel, label: str, is_used: bool = True) -> None:
        """
        Add a channel to the analyzer.
        
        Args:
            channel: Channel to add
            label: Label for the channel
            is_used: Whether the channel is used by the analyzer
        """
        self._cpp_settings.add_channel(channel, label, is_used)
        key = f"{channel.device_index}:{channel.channel_index}"
        self._channels[key] = (channel, label, is_used)
    
    def set_error_text(self, error_text: str) -> None:
        """
        Set error text for invalid settings.
        
        Args:
            error_text: Error message to display
        """
        self._cpp_settings.set_error_text(error_text)
    
    def add_export_option(self, user_id: int, menu_text: str) -> None:
        """
        Add an export option to the analyzer.
        
        Args:
            user_id: User-defined ID for the export option
            menu_text: Text to display in the export menu
        """
        self._cpp_settings.add_export_option(user_id, menu_text)
    
    def add_export_extension(self, user_id: int, extension_description: str, extension: str) -> None:
        """
        Add a file extension for an export option.
        
        Args:
            user_id: User-defined ID for the export option
            extension_description: Description of the file type
            extension: File extension (e.g., "csv")
        """
        self._cpp_settings.add_export_extension(user_id, extension_description, extension)
    
    @property
    def display_base(self) -> DisplayBase:
        """Get the display base used by the analyzer."""
        return self._cpp_settings.get_analyzer_display_base()
    
    @display_base.setter
    def display_base(self, base: DisplayBase) -> None:
        """Set the display base used by the analyzer."""
        self._cpp_settings.set_analyzer_display_base(base)
    
    @property
    def use_system_display_base(self) -> bool:
        """Check if the analyzer uses the system display base."""
        return self._cpp_settings.get_use_system_display_base()
    
    @use_system_display_base.setter
    def use_system_display_base(self, use_system: bool) -> None:
        """Set whether to use the system display base."""
        self._cpp_settings.set_use_system_display_base(use_system)
    
    def to_json(self) -> str:
        """
        Serialize settings to a JSON string.
        
        Returns:
            JSON string representing the settings
        """
        interfaces_data = []
        for interface in self._interfaces:
            interfaces_data.append(interface.to_dict())
        
        channels_data = {}
        for key, (channel, label, is_used) in self._channels.items():
            channels_data[key] = {
                "device_index": channel.device_index,
                "channel_index": channel.channel_index,
                "label": label,
                "is_used": is_used
            }
        
        data = {
            "interfaces": interfaces_data,
            "channels": channels_data,
            "display_base": int(self.display_base),
            "use_system_display_base": self.use_system_display_base
        }
        
        return json.dumps(data, indent=2)
    
    def from_json(self, json_str: str) -> None:
        """
        Deserialize settings from a JSON string.
        
        Args:
            json_str: JSON string representing the settings
        """
        data = json.loads(json_str)
        
        # Restore display settings
        if "display_base" in data:
            self.display_base = DisplayBase(data["display_base"])
        
        if "use_system_display_base" in data:
            self.use_system_display_base = data["use_system_display_base"]
        
        # This is a basic implementation - subclasses should override this method
        # to properly reconstruct their specific interfaces from the JSON data


class SimpleAnalyzerSettings(AnalyzerSettings):
    """
    A simplified implementation of AnalyzerSettings with basic functionality.
    
    This class provides a concrete implementation of AnalyzerSettings that
    can be used directly for simple analyzers without creating a custom subclass.
    
    Example:
        ```python
        # Create simple analyzer settings
        settings = SimpleAnalyzerSettings()
        
        # Add settings interfaces
        data_channel = ChannelSetting("Data Channel", "Select the data signal")
        clock_channel = ChannelSetting("Clock Channel", "Select the clock signal") 
        
        settings.add_interface(data_channel)
        settings.add_interface(clock_channel)
        
        # Configure channels
        settings.add_channel(Channel(0, 0), "Data", True)
        settings.add_channel(Channel(0, 1), "Clock", True)
        ```
    """
    
    def _create_cpp_settings(self) -> CppAnalyzerSettings:
        """Create a simple C++ settings implementation."""
        
        # Create a wrapper class that implements the pure virtual methods
        class SimpleSettings(CppAnalyzerSettings):
            def __init__(self, parent: SimpleAnalyzerSettings):
                super().__init__()
                self.parent = parent
                self._settings_str = ""
            
            def set_settings_from_interfaces(self) -> bool:
                # Simple implementation that always succeeds
                return True
            
            def load_settings(self, settings: str) -> None:
                self._settings_str = settings
                try:
                    self.parent.from_json(settings)
                except Exception:
                    # Ignore errors in deserialization
                    pass
            
            def save_settings(self) -> str:
                try:
                    self._settings_str = self.parent.to_json()
                except Exception:
                    # Fallback to empty JSON if serialization fails
                    self._settings_str = "{}"
                
                return self.set_return_string(self._settings_str)
        
        return SimpleSettings(self)


class SettingsBuilder:
    """
    Builder for fluently creating analyzer settings.
    
    This class provides a fluent interface for creating analyzer settings
    without explicitly creating each setting interface.
    
    Example:
        ```python
        # Create settings using the builder
        settings = (SettingsBuilder()
                    .add_channel("data", "Data Line", "Select the data signal channel")
                    .add_channel("clock", "Clock Line", "Select the clock signal channel") 
                    .add_number_list("baud", "Baud Rate", "Select the communication rate",
                                     options=[(9600, "9600 bps"), (115200, "115.2 kbps")])
                    .add_bool("invert", "Invert", "Invert signal polarity", checkbox_text="Invert signals")
                    .build())
        
        # Access settings by name
        data_channel = settings.get_setting("data").channel
        baud_rate = settings.get_setting("baud").value
        ```
    """
    
    def __init__(self):
        """Initialize a settings builder."""
        self._settings = SimpleAnalyzerSettings()
        self._interface_map: Dict[str, SettingInterface] = {}
    
    def add_channel(
        self, 
        name: str, 
        title: str, 
        tooltip: str = "", 
        channel: Optional[Channel] = None, 
        allow_none: bool = False
    ) -> 'SettingsBuilder':
        """
        Add a channel setting.
        
        Args:
            name: Name for accessing the setting
            title: Title/label for the setting
            tooltip: Tooltip text explaining the setting
            channel: Initial channel selection (optional)
            allow_none: Whether "None" is a valid selection
            
        Returns:
            Self for method chaining
        """
        setting = ChannelSetting(title, tooltip, channel, allow_none)
        self._settings.add_interface(setting)
        self._interface_map[name] = setting
        return self
    
    def add_number_list(
        self, 
        name: str, 
        title: str, 
        tooltip: str = "", 
        options: Optional[List[Union[float, Tuple[float, str, str]]]] = None
    ) -> 'SettingsBuilder':
        """
        Add a number list setting.
        
        Args:
            name: Name for accessing the setting
            title: Title/label for the setting
            tooltip: Tooltip text explaining the setting
            options: List of options to add. Can be:
                     - List of values
                     - List of (value, display_text, tooltip) tuples
                     
        Returns:
            Self for method chaining
        """
        setting = NumberListSetting(title, tooltip, options)
        self._settings.add_interface(setting)
        self._interface_map[name] = setting
        return self
    
    def add_integer(
        self, 
        name: str, 
        title: str, 
        tooltip: str = "", 
        value: int = 0, 
        min_value: int = 0, 
        max_value: int = 100
    ) -> 'SettingsBuilder':
        """
        Add an integer setting.
        
        Args:
            name: Name for accessing the setting
            title: Title/label for the setting
            tooltip: Tooltip text explaining the setting
            value: Initial value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Self for method chaining
        """
        setting = IntegerSetting(title, tooltip, value, min_value, max_value)
        self._settings.add_interface(setting)
        self._interface_map[name] = setting
        return self
    
    def add_text(
        self, 
        name: str, 
        title: str, 
        tooltip: str = "", 
        value: str = "", 
        field_type: TextFieldType = TextFieldType.TEXT
    ) -> 'SettingsBuilder':
        """
        Add a text setting.
        
        Args:
            name: Name for accessing the setting
            title: Title/label for the setting
            tooltip: Tooltip text explaining the setting
            value: Initial text value
            field_type: Type of text field (normal text, file path, or folder path)
            
        Returns:
            Self for method chaining
        """
        setting = TextSetting(title, tooltip, value, field_type)
        self._settings.add_interface(setting)
        self._interface_map[name] = setting
        return self
    
    def add_bool(
        self, 
        name: str, 
        title: str, 
        tooltip: str = "", 
        value: bool = False, 
        checkbox_text: str = ""
    ) -> 'SettingsBuilder':
        """
        Add a boolean setting.
        
        Args:
            name: Name for accessing the setting
            title: Title/label for the setting
            tooltip: Tooltip text explaining the setting
            value: Initial boolean value
            checkbox_text: Text to display next to the checkbox
            
        Returns:
            Self for method chaining
        """
        setting = BoolSetting(title, tooltip, value, checkbox_text)
        self._settings.add_interface(setting)
        self._interface_map[name] = setting
        return self
    
    def build(self) -> 'BuilderSettings':
        """
        Build the settings object.
        
        Returns:
            A BuilderSettings object containing all added settings
        """
        return BuilderSettings(self._settings, self._interface_map)


class BuilderSettings(SimpleAnalyzerSettings):
    """
    Settings object created by the SettingsBuilder.
    
    This class extends SimpleAnalyzerSettings with the ability to access
    settings by name.
    """
    
    def __init__(self, settings: SimpleAnalyzerSettings, interface_map: Dict[str, SettingInterface]):
        """
        Initialize builder settings.
        
        Args:
            settings: Base settings object
            interface_map: Map of setting names to interfaces
        """
        # Take ownership of the C++ settings object from the base settings
        self._cpp_settings = settings.cpp_settings
        
        # Copy interfaces and channels from base settings
        self._interfaces = settings._interfaces
        self._channels = settings._channels
        
        # Store the interface map
        self._interface_map = interface_map
    
    def _create_cpp_settings(self) -> CppAnalyzerSettings:
        """Not used - we take ownership of the existing C++ settings object."""
        raise NotImplementedError("BuilderSettings does not create a new C++ settings object")
    
    def get_setting(self, name: str) -> SettingInterface:
        """
        Get a setting interface by name.
        
        Args:
            name: Name of the setting
            
        Returns:
            The setting interface
            
        Raises:
            KeyError: If no setting with the given name exists
        """
        if name not in self._interface_map:
            raise KeyError(f"No setting named '{name}'")
        return self._interface_map[name]
    
    def __getattr__(self, name: str) -> SettingInterface:
        """
        Access settings as attributes.
        
        This allows settings to be accessed using dot notation:
        settings.data_channel instead of settings.get_setting("data_channel")
        
        Args:
            name: Name of the setting
            
        Returns:
            The setting interface
            
        Raises:
            AttributeError: If no setting with the given name exists
        """
        try:
            return self.get_setting(name)
        except KeyError:
            raise AttributeError(f"'BuilderSettings' object has no attribute '{name}'")


# Helper functions for common settings configurations

def create_common_baud_rates() -> List[Tuple[float, str, str]]:
    """
    Create a list of common baud rates for serial communication.
    
    Returns:
        List of (rate, display_text, tooltip) tuples
    """
    return [
        (1200, "1200 bps", "Low speed"),
        (2400, "2400 bps", "Low speed"),
        (4800, "4800 bps", "Low speed"),
        (9600, "9600 bps", "Standard speed"),
        (19200, "19200 bps", "Medium speed"),
        (38400, "38400 bps", "Medium speed"),
        (57600, "57600 bps", "High speed"),
        (115200, "115.2 kbps", "High speed"),
        (230400, "230.4 kbps", "Very high speed"),
        (460800, "460.8 kbps", "Very high speed"),
        (921600, "921.6 kbps", "Very high speed"),
        (1000000, "1 Mbps", "Maximum speed")
    ]


def create_common_bit_counts() -> List[Tuple[float, str, str]]:
    """
    Create a list of common bit counts for data words.
    
    Returns:
        List of (count, display_text, tooltip) tuples
    """
    return [
        (5, "5 bits", "5-bit data word"),
        (6, "6 bits", "6-bit data word"),
        (7, "7 bits", "7-bit data word (ASCII)"),
        (8, "8 bits", "8-bit data word (standard byte)"),
        (9, "9 bits", "9-bit data word (with parity)"),
        (10, "10 bits", "10-bit data word"),
        (16, "16 bits", "16-bit data word")
    ]


def create_uart_settings() -> BuilderSettings:
    """
    Create settings for a UART (serial) analyzer.
    
    Returns:
        Preconfigured settings for UART analysis
    """
    builder = SettingsBuilder()
    
    # Add standard UART settings
    builder.add_channel("tx", "TX Channel", "Transmit data line")
    builder.add_channel("rx", "RX Channel", "Receive data line", allow_none=True)
    
    # Add baud rate setting
    builder.add_number_list("baud_rate", "Baud Rate", "Serial communication rate", 
                           create_common_baud_rates())
    
    # Add data bits setting
    builder.add_number_list("bits", "Data Bits", "Number of data bits per word",
                           [(5, "5 bits"), (6, "6 bits"), (7, "7 bits"), (8, "8 bits")])
    
    # Add parity setting
    builder.add_number_list("parity", "Parity", "Error checking method",
                           [(0, "None", "No parity bit"),
                            (1, "Odd", "Odd parity checking"),
                            (2, "Even", "Even parity checking")])
    
    # Add stop bits setting
    builder.add_number_list("stop_bits", "Stop Bits", "Number of stop bits",
                           [(1, "1 bit"), (1.5, "1.5 bits"), (2, "2 bits")])
    
    # Add invert setting
    builder.add_bool("invert", "Invert", "Invert signal polarity", 
                    checkbox_text="Invert (RS-232 logic)")
    
    # Set defaults
    settings = builder.build()
    settings.baud_rate.value = 9600
    settings.bits.value = 8
    settings.parity.value = 0
    settings.stop_bits.value = 1
    settings.invert.value = False
    
    return settings


def create_spi_settings() -> BuilderSettings:
    """
    Create settings for a SPI analyzer.
    
    Returns:
        Preconfigured settings for SPI analysis
    """
    builder = SettingsBuilder()
    
    # Add standard SPI settings
    builder.add_channel("mosi", "MOSI", "Master Out Slave In data line", allow_none=True)
    builder.add_channel("miso", "MISO", "Master In Slave Out data line", allow_none=True)
    builder.add_channel("clock", "Clock", "Clock signal")
    builder.add_channel("enable", "Enable", "Chip select/enable signal", allow_none=True)
    
    # Add shift order setting
    builder.add_number_list("shift_order", "Shift Order", "Bit order for data transmission",
                           [(0, "MSB First", "Most significant bit first"),
                            (1, "LSB First", "Least significant bit first")])
    
    # Add clock polarity setting
    builder.add_number_list("clock_polarity", "Clock Polarity", "Idle state of the clock signal",
                           [(0, "CPOL=0 (Idle Low)", "Clock is low when idle"),
                            (1, "CPOL=1 (Idle High)", "Clock is high when idle")])
    
    # Add clock phase setting
    builder.add_number_list("clock_phase", "Clock Phase", "When data is sampled",
                           [(0, "CPHA=0 (Sample Leading)", "Data sampled on leading edge"),
                            (1, "CPHA=1 (Sample Trailing)", "Data sampled on trailing edge")])
    
    # Add data bits setting
    builder.add_number_list("bits", "Bits per Transfer", "Number of bits per word",
                           [(8, "8 bits"), (16, "16 bits"), (24, "24 bits"), (32, "32 bits")])
    
    # Add enable polarity setting
    builder.add_number_list("enable_polarity", "Enable Polarity", "Active state of enable signal",
                           [(0, "Active Low"), (1, "Active High")])
    
    # Set defaults
    settings = builder.build()
    settings.shift_order.value = 0  # MSB First
    settings.clock_polarity.value = 0  # CPOL=0
    settings.clock_phase.value = 0  # CPHA=0
    settings.bits.value = 8  # 8 bits
    settings.enable_polarity.value = 0  # Active Low
    
    return settings


def create_i2c_settings() -> BuilderSettings:
    """
    Create settings for an I2C analyzer.
    
    Returns:
        Preconfigured settings for I2C analysis
    """
    builder = SettingsBuilder()
    
    # Add standard I2C settings
    builder.add_channel("sda", "SDA", "Serial Data line")
    builder.add_channel("scl", "SCL", "Serial Clock line")
    
    # Add address bits setting
    builder.add_number_list("address_bits", "Address Bits", "Number of bits in slave address",
                           [(7, "7 bits (Standard)"), (10, "10 bits (Extended)")])
    
    # Add display options setting
    builder.add_bool("show_ack", "Show ACK/NACK", "Display acknowledge/not-acknowledge bits",
                    checkbox_text="Show ACK/NACK")
    
    # Add timing analysis setting
    builder.add_bool("timing_analysis", "Timing Analysis", "Analyze timing violations",
                    checkbox_text="Analyze timing violations")
    
    # Set defaults
    settings = builder.build()
    settings.address_bits.value = 7  # 7-bit addressing
    settings.show_ack.value = True
    settings.timing_analysis.value = False
    
    return settings
