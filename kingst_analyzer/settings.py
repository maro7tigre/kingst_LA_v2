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
    uart_settings = create_uart_settings()
    uart_settings.tx.channel = Channel(0, 0)
    uart_settings.baud_rate.value = 115200
    
    # Using builder pattern
    settings = (SettingsBuilder()
                .add_channel("data", "Data Line")
                .add_channel("clock", "Clock Line")
                .add_number_list("baud", "Baud Rate", options=[9600, 115200])
                .build())
                
    # Creating custom protocol settings
    class MyProtocolSettings(AnalyzerSettings):
        def __init__(self):
            super().__init__()
            self._create_interfaces()
            self._register_interfaces()
            
        def _create_interfaces(self):
            self.clock = ChannelSetting("Clock", "Clock signal line")
            self.data = ChannelSetting("Data", "Data signal line")
            self.rate = NumberListSetting("Rate", "Transfer rate")
            
        def _register_interfaces(self):
            self.add_interface(self.clock)
            self.add_interface(self.data)
            self.add_interface(self.rate)
            
        def validate(self) -> Tuple[bool, str]:
            # Custom validation logic
            return True, ""
"""

from typing import List, Dict, Union, Optional, Tuple, Any, Callable, TypeVar, Generic, cast, Set, Iterator
import json
import abc
from enum import Enum, auto
from dataclasses import dataclass
import warnings

# Import the C++ bound classes
try:
    from kingst_analyzer import (
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
except ImportError:
    # For development/documentation without the actual bindings
    class AnalyzerInterfaceTypeId(Enum):
        INTERFACE_BASE = 0
        INTERFACE_CHANNEL = 1
        INTERFACE_NUMBER_LIST = 2
        INTERFACE_INTEGER = 3
        INTERFACE_TEXT = 4
        INTERFACE_BOOL = 5
    
    class Channel:
        """Stub class for documentation"""
        def __init__(self, device_index: int, channel_index: int):
            self.device_index = device_index
            self.channel_index = channel_index
    
    class DisplayBase(Enum):
        ASCII = 0
        DECIMAL = 1
        HEXADECIMAL = 2
        BINARY = 3
        OCT = 4
    
    class AnalyzerSettingInterface:
        """Stub class for documentation"""
        pass
        
    class AnalyzerSettingInterfaceChannel(AnalyzerSettingInterface):
        """Stub class for documentation"""
        pass
        
    class AnalyzerSettingInterfaceNumberList(AnalyzerSettingInterface):
        """Stub class for documentation"""
        pass
        
    class AnalyzerSettingInterfaceInteger(AnalyzerSettingInterface):
        """Stub class for documentation"""
        pass
        
    class AnalyzerSettingInterfaceText(AnalyzerSettingInterface):
        """Stub class for documentation"""
        class TextType(Enum):
            NormalText = 0
            FilePath = 1
            FolderPath = 2
            
    class AnalyzerSettingInterfaceBool(AnalyzerSettingInterface):
        """Stub class for documentation"""
        pass
        
    class CppAnalyzerSettings:
        """Stub class for documentation"""
        pass

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
    'BuilderSettings',
    'TextFieldType',
    'validate_settings',
    'create_uart_settings',
    'create_spi_settings',
    'create_i2c_settings',
    'create_common_baud_rates',
    'create_common_bit_counts',
    'SettingsValidationError',
]

# Type variable for generic settings
T = TypeVar('T')


class TextFieldType(Enum):
    """
    Types of text input fields.
    
    This enum defines the different types of text fields that can be used in a text setting.
    Each type affects how the UI displays and handles the text input.
    
    Attributes:
        TEXT: Regular text input field
        FILE_PATH: File path input with browse button
        FOLDER_PATH: Folder path input with browse button
    """
    TEXT = AnalyzerSettingInterfaceText.TextType.NormalText
    FILE_PATH = AnalyzerSettingInterfaceText.TextType.FilePath
    FOLDER_PATH = AnalyzerSettingInterfaceText.TextType.FolderPath


class SettingsValidationError(Exception):
    """
    Exception raised when settings validation fails.
    
    This exception is used to indicate that the settings validation process
    failed, with a message explaining why.
    
    Attributes:
        message: Explanation of the validation error
    """
    
    def __init__(self, message: str):
        """
        Initialize a settings validation error.
        
        Args:
            message: Explanation of the validation error
        """
        self.message = message
        super().__init__(self.message)


def validate_settings(func: Callable) -> Callable:
    """
    Decorator for methods that should validate settings before proceeding.
    
    This decorator wraps a method to ensure that settings are valid before
    the method is executed.
    
    Example:
        ```python
        class MySettings(AnalyzerSettings):
            @validate_settings
            def apply_settings(self):
                # This will only run if settings are valid
                pass
        ```
    
    Args:
        func: The method to wrap
        
    Returns:
        Wrapped method that validates settings before execution
    """
    def wrapper(self, *args, **kwargs):
        if not self.is_valid():
            raise SettingsValidationError(self.get_error_message())
        return func(self, *args, **kwargs)
    return wrapper


class SettingInterface(abc.ABC):
    """
    Base class for all settings interfaces.
    
    This abstract class defines the common properties and methods that all
    setting interfaces should implement, providing a consistent API. Each
    specific setting type (channel, number list, integer, text, boolean)
    inherits from this base class.
    
    Attributes:
        title (str): Title/label for the setting
        tooltip (str): Tooltip text explaining the setting
        disabled (bool): Whether the setting is disabled
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
        """
        Get the underlying C++ interface object.
        
        Returns:
            The C++ interface object
            
        Raises:
            RuntimeError: If the C++ interface is not initialized
        """
        if self._cpp_interface is None:
            raise RuntimeError("C++ interface not initialized")
        return self._cpp_interface
    
    @property
    def title(self) -> str:
        """
        Get the title/label for this setting.
        
        Returns:
            Title text
        """
        if self._cpp_interface is None:
            return self._title
        return self._cpp_interface.get_title()
    
    @title.setter
    def title(self, value: str) -> None:
        """
        Set the title/label for this setting.
        
        Args:
            value: Title text
        """
        self._title = value
        if self._cpp_interface is not None:
            self._cpp_interface.set_title_and_tooltip(value, self.tooltip)
    
    @property
    def tooltip(self) -> str:
        """
        Get the tooltip text for this setting.
        
        Returns:
            Tooltip text
        """
        if self._cpp_interface is None:
            return self._tooltip
        return self._cpp_interface.get_tool_tip()
    
    @tooltip.setter
    def tooltip(self, value: str) -> None:
        """
        Set the tooltip text for this setting.
        
        Args:
            value: Tooltip text
        """
        self._tooltip = value
        if self._cpp_interface is not None:
            self._cpp_interface.set_title_and_tooltip(self.title, value)
    
    @property
    def disabled(self) -> bool:
        """
        Check if this setting is disabled.
        
        Returns:
            True if the setting is disabled
        """
        if self._cpp_interface is None:
            return False
        return self._cpp_interface.is_disabled()
    
    def enable(self) -> None:
        """
        Enable this setting.
        
        This method is not directly available in the C++ API, but is provided
        for symmetry with the disable method. In the C++ API, settings are
        enabled by default.
        """
        # This functionality is not directly available in the C++ API
        # Settings are enabled by default
        pass
    
    def disable(self) -> None:
        """
        Disable this setting.
        
        This method is not directly available in the C++ API. In a real
        implementation, this would set a flag to disable the setting.
        """
        # This functionality is not directly available in the C++ API
        # In a real implementation, this would set a flag to disable the setting
        pass
    
    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the setting to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the setting
        """
        return {
            "type": self.__class__.__name__,
            "title": self.title,
            "tooltip": self.tooltip,
        }
    
    @classmethod
    @abc.abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SettingInterface':
        """
        Create a setting from a dictionary.
        
        Args:
            data: Dictionary representation of the setting
            
        Returns:
            New setting instance
        """
        raise NotImplementedError
    
    def __str__(self) -> str:
        """
        Get a string representation of the setting.
        
        Returns:
            String representation
        """
        return f"{self.__class__.__name__}(title='{self.title}')"
    
    def __repr__(self) -> str:
        """
        Get a detailed string representation of the setting.
        
        Returns:
            Detailed string representation
        """
        return f"{self.__class__.__name__}(title='{self.title}', tooltip='{self.tooltip}')"


class ChannelSetting(SettingInterface):
    """
    Setting for selecting a channel.
    
    This interface allows the user to select a channel from the available channels.
    
    Attributes:
        channel (Channel): The currently selected channel
        allow_none (bool): Whether "None" is a valid selection
        
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
        """
        Get the currently selected channel.
        
        Returns:
            The selected channel
        """
        return cast(AnalyzerSettingInterfaceChannel, self.cpp_interface).get_channel()
    
    @channel.setter
    def channel(self, value: Channel) -> None:
        """
        Set the selected channel.
        
        Args:
            value: The channel to select
        """
        cast(AnalyzerSettingInterfaceChannel, self.cpp_interface).set_channel(value)
    
    @property
    def allow_none(self) -> bool:
        """
        Check if selecting "None" is allowed for this channel setting.
        
        Returns:
            True if "None" selection is allowed
        """
        return cast(AnalyzerSettingInterfaceChannel, self.cpp_interface).get_selection_of_none_is_allowed()
    
    @allow_none.setter
    def allow_none(self, value: bool) -> None:
        """
        Set whether selecting "None" is allowed for this channel setting.
        
        Args:
            value: True to allow "None" selection
        """
        cast(AnalyzerSettingInterfaceChannel, self.cpp_interface).set_selection_of_none_is_allowed(value)
    
    def is_valid(self) -> bool:
        """
        Check if the current channel selection is valid.
        
        Returns:
            True if the selection is valid
        """
        # If None is not allowed, then we need to have a valid channel
        if not self.allow_none:
            channel = self.channel
            return channel.device_index >= 0 and channel.channel_index >= 0
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the setting to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the setting
        """
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
        """
        Create a channel setting from a dictionary.
        
        Args:
            data: Dictionary representation of the setting
            
        Returns:
            New channel setting instance
        """
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
    
    def __str__(self) -> str:
        """
        Get a string representation of the channel setting.
        
        Returns:
            String representation
        """
        channel = self.channel
        channel_str = f"({channel.device_index}, {channel.channel_index})" if channel else "None"
        return f"{self.__class__.__name__}(title='{self.title}', channel={channel_str})"


class NumberListOption:
    """
    Represents an option in a number list setting.
    
    This class encapsulates a numeric option with associated display text and tooltip.
    
    Attributes:
        value (float): The numeric value
        display_text (str): Text to display for this option
        tooltip (str): Tooltip for this option
    """
    
    def __init__(self, value: float, display_text: str, tooltip: str = ""):
        """
        Initialize a number list option.
        
        Args:
            value: Numeric value
            display_text: Display text for the option
            tooltip: Tooltip for the option
        """
        self.value = value
        self.display_text = display_text
        self.tooltip = tooltip
    
    def __str__(self) -> str:
        """
        Get a string representation of the option.
        
        Returns:
            String representation
        """
        return f"{self.display_text} ({self.value})"
    
    def __repr__(self) -> str:
        """
        Get a detailed string representation of the option.
        
        Returns:
            Detailed string representation
        """
        return f"NumberListOption(value={self.value}, display_text='{self.display_text}', tooltip='{self.tooltip}')"


class NumberListSetting(SettingInterface):
    """
    Setting for selecting from a list of numeric values.
    
    This interface is used for settings where the user selects from a list of options,
    each with an associated numeric value (e.g., baud rates, bit counts, etc.).
    
    Attributes:
        value (float): The currently selected numeric value
        
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
        
        # Iterate through all options
        for option in baud_setting.options:
            print(f"Option: {option.display_text}, Value: {option.value}")
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
        self._options: List[NumberListOption] = []
        
        if options:
            for option in options:
                if isinstance(option, tuple):
                    value, text, option_tooltip = option
                    self.add_option(value, text, option_tooltip)
                else:
                    self.add_option(option, str(option), "")
    
    @property
    def value(self) -> float:
        """
        Get the currently selected value.
        
        Returns:
            The selected value
        """
        return cast(AnalyzerSettingInterfaceNumberList, self.cpp_interface).get_number()
    
    @value.setter
    def value(self, number: float) -> None:
        """
        Set the selected value.
        
        Args:
            number: Value to select
            
        Raises:
            ValueError: If the value is not in the list of options
        """
        # Check if the value is in the list of options
        values = [option.value for option in self.options]
        if number not in values and values:
            raise ValueError(f"Value {number} is not in the list of options: {values}")
            
        cast(AnalyzerSettingInterfaceNumberList, self.cpp_interface).set_number(number)
    
    @property
    def options(self) -> List[NumberListOption]:
        """
        Get all options in the list.
        
        Returns:
            List of all options
        """
        # Update our cached options list from the C++ interface
        interface = cast(AnalyzerSettingInterfaceNumberList, self.cpp_interface)
        count = interface.get_listbox_numbers_count()
        
        # Clear and rebuild the options list
        self._options = []
        
        for i in range(count):
            value = interface.get_listbox_number(i)
            text = interface.get_listbox_string(i)
            tooltip = ""
            if i < interface.get_listbox_tooltips_count():
                tooltip = interface.get_listbox_tooltip(i)
            self._options.append(NumberListOption(value, text, tooltip))
        
        return self._options
    
    def add_option(self, value: float, display_text: str, tooltip: str = "") -> None:
        """
        Add an option to the list.
        
        Args:
            value: Numeric value
            display_text: Display text for the option
            tooltip: Tooltip for the option
        """
        cast(AnalyzerSettingInterfaceNumberList, self.cpp_interface).add_number(value, display_text, tooltip)
        self._options.append(NumberListOption(value, display_text, tooltip))
    
    def clear_options(self) -> None:
        """
        Clear all options from the list.
        """
        cast(AnalyzerSettingInterfaceNumberList, self.cpp_interface).clear_numbers()
        self._options = []
    
    def get_option_by_value(self, value: float) -> Optional[NumberListOption]:
        """
        Find an option by its value.
        
        Args:
            value: Value to search for
            
        Returns:
            The option with the specified value, or None if not found
        """
        for option in self.options:
            if option.value == value:
                return option
        return None
    
    def get_option_by_text(self, text: str) -> Optional[NumberListOption]:
        """
        Find an option by its display text.
        
        Args:
            text: Display text to search for
            
        Returns:
            The option with the specified display text, or None if not found
        """
        for option in self.options:
            if option.display_text == text:
                return option
        return None
    
    def is_valid(self) -> bool:
        """
        Check if the current value is valid.
        
        A value is valid if the options list is empty (any value is allowed)
        or if the value is in the list of options.
        
        Returns:
            True if the value is valid
        """
        values = [option.value for option in self.options]
        if not values:
            return True
        return self.value in values
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the setting to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the setting
        """
        data = super().to_dict()
        
        # Convert options to a serializable format
        options_data = []
        for option in self.options:
            options_data.append({
                "value": option.value,
                "display_text": option.display_text,
                "tooltip": option.tooltip
            })
        
        data.update({
            "options": options_data,
            "value": self.value
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NumberListSetting':
        """
        Create a number list setting from a dictionary.
        
        Args:
            data: Dictionary representation of the setting
            
        Returns:
            New number list setting instance
        """
        setting = cls(
            title=data.get("title", ""),
            tooltip=data.get("tooltip", ""),
        )
        
        # Add options
        options_data = data.get("options", [])
        for option_data in options_data:
            if isinstance(option_data, dict):
                setting.add_option(
                    option_data["value"],
                    option_data["display_text"],
                    option_data.get("tooltip", "")
                )
            elif isinstance(option_data, (list, tuple)):
                if len(option_data) >= 3:
                    setting.add_option(option_data[0], option_data[1], option_data[2])
                elif len(option_data) >= 2:
                    setting.add_option(option_data[0], option_data[1])
                elif len(option_data) >= 1:
                    setting.add_option(option_data[0], str(option_data[0]))
            else:
                setting.add_option(option_data, str(option_data))
        
        # Set the value
        if "value" in data:
            try:
                setting.value = data["value"]
            except ValueError:
                # If the value is not in the list of options, ignore it
                pass
            
        return setting
    
    def __str__(self) -> str:
        """
        Get a string representation of the number list setting.
        
        Returns:
            String representation
        """
        return f"{self.__class__.__name__}(title='{self.title}', value={self.value}, options_count={len(self.options)})"


class IntegerSetting(SettingInterface):
    """
    Setting for entering an integer value.
    
    This interface allows the user to enter an integer value within a specified range.
    
    Attributes:
        value (int): The current integer value
        min_value (int): The minimum allowed value
        max_value (int): The maximum allowed value
        
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
        
        # Set range before setting value
        self._cpp_interface.set_min(min_value)
        self._cpp_interface.set_max(max_value)
        
        # Set initial value
        try:
            self._cpp_interface.set_integer(value)
        except Exception:
            # If the initial value is out of range, set to min_value
            self._cpp_interface.set_integer(min_value)
    
    @property
    def value(self) -> int:
        """
        Get the current integer value.
        
        Returns:
            Current value
        """
        return cast(AnalyzerSettingInterfaceInteger, self.cpp_interface).get_integer()
    
    @value.setter
    def value(self, integer: int) -> None:
        """
        Set the integer value.
        
        Args:
            integer: Value to set
            
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
        """
        Get the minimum allowed value.
        
        Returns:
            Minimum value
        """
        return cast(AnalyzerSettingInterfaceInteger, self.cpp_interface).get_min()
    
    @min_value.setter
    def min_value(self, value: int) -> None:
        """
        Set the minimum allowed value.
        
        Args:
            value: Minimum value
            
        Raises:
            ValueError: If the minimum value is greater than the maximum value
        """
        interface = cast(AnalyzerSettingInterfaceInteger, self.cpp_interface)
        max_val = interface.get_max()
        
        if value > max_val:
            raise ValueError(f"Minimum value ({value}) cannot be greater than maximum value ({max_val})")
            
        interface.set_min(value)
        
        # If the current value is less than the new minimum, adjust it
        current = interface.get_integer()
        if current < value:
            interface.set_integer(value)
    
    @property
    def max_value(self) -> int:
        """
        Get the maximum allowed value.
        
        Returns:
            Maximum value
        """
        return cast(AnalyzerSettingInterfaceInteger, self.cpp_interface).get_max()
    
    @max_value.setter
    def max_value(self, value: int) -> None:
        """
        Set the maximum allowed value.
        
        Args:
            value: Maximum value
            
        Raises:
            ValueError: If the maximum value is less than the minimum value
        """
        interface = cast(AnalyzerSettingInterfaceInteger, self.cpp_interface)
        min_val = interface.get_min()
        
        if value < min_val:
            raise ValueError(f"Maximum value ({value}) cannot be less than minimum value ({min_val})")
            
        interface.set_max(value)
        
        # If the current value is greater than the new maximum, adjust it
        current = interface.get_integer()
        if current > value:
            interface.set_integer(value)
    
    def is_valid(self) -> bool:
        """
        Check if the current value is valid.
        
        A value is valid if it is within the allowed range.
        
        Returns:
            True if the value is valid
        """
        return self.min_value <= self.value <= self.max_value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the setting to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the setting
        """
        data = super().to_dict()
        data.update({
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntegerSetting':
        """
        Create an integer setting from a dictionary.
        
        Args:
            data: Dictionary representation of the setting
            
        Returns:
            New integer setting instance
        """
        setting = cls(
            title=data.get("title", ""),
            tooltip=data.get("tooltip", ""),
            min_value=data.get("min_value", 0),
            max_value=data.get("max_value", 100)
        )
        
        # Set value after creating setting to ensure it's within range
        if "value" in data:
            try:
                setting.value = data["value"]
            except ValueError:
                # If the value is out of range, leave at default (already set by constructor)
                pass
            
        return setting
    
    def __str__(self) -> str:
        """
        Get a string representation of the integer setting.
        
        Returns:
            String representation
        """
        return f"{self.__class__.__name__}(title='{self.title}', value={self.value}, range={self.min_value}-{self.max_value})"


class TextSetting(SettingInterface):
    """
    Setting for entering text.
    
    This interface allows the user to enter text, which can be used for various purposes
    such as naming, entering file paths, etc.
    
    Attributes:
        value (str): The current text value
        field_type (TextFieldType): The type of text field
        
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
        """
        Get the current text value.
        
        Returns:
            Current text
        """
        return cast(AnalyzerSettingInterfaceText, self.cpp_interface).get_text()
    
    @value.setter
    def value(self, text: str) -> None:
        """
        Set the text value.
        
        Args:
            text: Text to set
        """
        cast(AnalyzerSettingInterfaceText, self.cpp_interface).set_text(text)
    
    @property
    def field_type(self) -> TextFieldType:
        """
        Get the type of text field.
        
        Returns:
            The text field type
        """
        cpp_type = cast(AnalyzerSettingInterfaceText, self.cpp_interface).get_text_type()
        return TextFieldType(cpp_type)
    
    @field_type.setter
    def field_type(self, field_type: TextFieldType) -> None:
        """
        Set the type of text field.
        
        Args:
            field_type: The text field type
        """
        cast(AnalyzerSettingInterfaceText, self.cpp_interface).set_text_type(field_type.value)
    
    def is_valid(self) -> bool:
        """
        Check if the current text value is valid.
        
        This implementation always returns True, as there are no specific
        validation rules for text fields. Subclasses can override this method
        to implement specific validation rules.
        
        Returns:
            True if the text is valid
        """
        # No specific validation rules for text fields
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the setting to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the setting
        """
        data = super().to_dict()
        data.update({
            "value": self.value,
            "field_type": self.field_type.name
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TextSetting':
        """
        Create a text setting from a dictionary.
        
        Args:
            data: Dictionary representation of the setting
            
        Returns:
            New text setting instance
        """
        field_type = TextFieldType.TEXT
        if "field_type" in data:
            try:
                field_type = TextFieldType[data["field_type"]]
            except KeyError:
                # If the field type is not valid, use default
                pass
            
        return cls(
            title=data.get("title", ""),
            tooltip=data.get("tooltip", ""),
            value=data.get("value", ""),
            field_type=field_type
        )
    
    def __str__(self) -> str:
        """
        Get a string representation of the text setting.
        
        Returns:
            String representation
        """
        return f"{self.__class__.__name__}(title='{self.title}', value='{self.value}', field_type={self.field_type.name})"


class BoolSetting(SettingInterface):
    """
    Setting for boolean (checkbox) values.
    
    This interface presents a checkbox with a label for toggling boolean options.
    
    Attributes:
        value (bool): The current boolean value
        checkbox_text (str): Text displayed next to the checkbox
        
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
        """
        Get the current boolean value.
        
        Returns:
            Current value
        """
        return cast(AnalyzerSettingInterfaceBool, self.cpp_interface).get_value()
    
    @value.setter
    def value(self, value: bool) -> None:
        """
        Set the boolean value.
        
        Args:
            value: Value to set
        """
        cast(AnalyzerSettingInterfaceBool, self.cpp_interface).set_value(value)
    
    @property
    def checkbox_text(self) -> str:
        """
        Get the text displayed next to the checkbox.
        
        Returns:
            Checkbox text
        """
        return cast(AnalyzerSettingInterfaceBool, self.cpp_interface).get_check_box_text()
    
    @checkbox_text.setter
    def checkbox_text(self, text: str) -> None:
        """
        Set the text displayed next to the checkbox.
        
        Args:
            text: Checkbox text
        """
        cast(AnalyzerSettingInterfaceBool, self.cpp_interface).set_check_box_text(text)
    
    def is_valid(self) -> bool:
        """
        Check if the current boolean value is valid.
        
        This implementation always returns True, as there are no specific
        validation rules for boolean fields.
        
        Returns:
            True if the value is valid
        """
        # Boolean values are always valid
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the setting to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the setting
        """
        data = super().to_dict()
        data.update({
            "value": self.value,
            "checkbox_text": self.checkbox_text
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BoolSetting':
        """
        Create a boolean setting from a dictionary.
        
        Args:
            data: Dictionary representation of the setting
            
        Returns:
            New boolean setting instance
        """
        return cls(
            title=data.get("title", ""),
            tooltip=data.get("tooltip", ""),
            value=data.get("value", False),
            checkbox_text=data.get("checkbox_text", "")
        )
    
    def __str__(self) -> str:
        """
        Get a string representation of the boolean setting.
        
        Returns:
            String representation
        """
        return f"{self.__class__.__name__}(title='{self.title}', value={self.value}, checkbox_text='{self.checkbox_text}')"


class AnalyzerSettings(abc.ABC):
    """
    Base class for analyzer settings.
    
    This abstract class provides a Pythonic interface to the C++ AnalyzerSettings class.
    It manages the settings interfaces, load/save functionality, and validation.
    
    Must be subclassed to implement specific analyzer settings.
    
    Attributes:
        display_base (DisplayBase): The display base used by the analyzer
        use_system_display_base (bool): Whether to use the system display base
        
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
        self._error_message: str = ""
    
    @property
    def cpp_settings(self) -> CppAnalyzerSettings:
        """
        Get the underlying C++ settings object.
        
        Returns:
            The C++ settings object
        """
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
    
    def get_interfaces(self) -> List[SettingInterface]:
        """
        Get all setting interfaces.
        
        Returns:
            List of all setting interfaces
        """
        return self._interfaces.copy()
    
    def get_interface_by_title(self, title: str) -> Optional[SettingInterface]:
        """
        Find a setting interface by its title.
        
        Args:
            title: Title to search for
            
        Returns:
            The setting interface with the specified title, or None if not found
        """
        for interface in self._interfaces:
            if interface.title == title:
                return interface
        return None
    
    def clear_channels(self) -> None:
        """
        Clear all reported channels.
        
        Call this before adding channels if the channel configuration changes.
        """
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
    
    def get_channels(self) -> Dict[str, Tuple[Channel, str, bool]]:
        """
        Get all channels used by the analyzer.
        
        Returns:
            Dictionary mapping channel keys to (channel, label, is_used) tuples
        """
        return self._channels.copy()
    
    def get_channel_by_label(self, label: str) -> Optional[Tuple[Channel, str, bool]]:
        """
        Find a channel by its label.
        
        Args:
            label: Label to search for
            
        Returns:
            The (channel, label, is_used) tuple with the specified label, or None if not found
        """
        for key, (channel, chan_label, is_used) in self._channels.items():
            if chan_label == label:
                return (channel, chan_label, is_used)
        return None
    
    def set_error_text(self, error_text: str) -> None:
        """
        Set error text for invalid settings.
        
        Args:
            error_text: Error message to display
        """
        self._error_message = error_text
        self._cpp_settings.set_error_text(error_text)
    
    def get_error_message(self) -> str:
        """
        Get the error message for invalid settings.
        
        Returns:
            Error message
        """
        return self._error_message
    
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
        """
        Get the display base used by the analyzer.
        
        Returns:
            Current display base
        """
        return self._cpp_settings.get_analyzer_display_base()
    
    @display_base.setter
    def display_base(self, base: DisplayBase) -> None:
        """
        Set the display base used by the analyzer.
        
        Args:
            base: Display base to use
        """
        self._cpp_settings.set_analyzer_display_base(base)
    
    @property
    def use_system_display_base(self) -> bool:
        """
        Check if the analyzer uses the system display base.
        
        Returns:
            True if using system display base
        """
        return self._cpp_settings.get_use_system_display_base()
    
    @use_system_display_base.setter
    def use_system_display_base(self, use_system: bool) -> None:
        """
        Set whether to use the system display base.
        
        Args:
            use_system: True to use system display base
        """
        self._cpp_settings.set_use_system_display_base(use_system)
    
    def is_valid(self) -> bool:
        """
        Check if all settings are valid.
        
        Returns:
            True if all settings are valid
        """
        # Check if all interfaces are valid
        for interface in self._interfaces:
            if hasattr(interface, 'is_valid') and not interface.is_valid():
                interface_name = interface.title or str(interface)
                self.set_error_text(f"Invalid setting: {interface_name}")
                return False
                
        # Custom validation by derived class
        valid, message = self.validate()
        if not valid:
            self.set_error_text(message)
            
        return valid
    
    def validate(self) -> Tuple[bool, str]:
        """
        Perform custom validation of settings.
        
        This method should be overridden by derived classes to perform
        validation that involves multiple settings or complex rules.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Default implementation - no custom validation
        return True, ""
    
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
    
    def __str__(self) -> str:
        """
        Get a string representation of the analyzer settings.
        
        Returns:
            String representation
        """
        return f"{self.__class__.__name__}(interfaces={len(self._interfaces)}, channels={len(self._channels)})"


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
                return self.parent.is_valid()
            
            def load_settings(self, settings: str) -> None:
                self._settings_str = settings
                try:
                    self.parent.from_json(settings)
                except Exception as e:
                    # Set error message for deserialization errors
                    self.parent.set_error_text(f"Error loading settings: {str(e)}")
            
            def save_settings(self) -> str:
                try:
                    self._settings_str = self.parent.to_json()
                except Exception as e:
                    # Fallback to empty JSON if serialization fails
                    self._settings_str = "{}"
                    self.parent.set_error_text(f"Error saving settings: {str(e)}")
                
                return self.set_return_string(self._settings_str)
            
            def get_setting_brief(self) -> str:
                # Generate a brief description of the settings
                channel_count = self.parent._channels.count()
                return f"Settings with {channel_count} channels"
        
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
    
    def add_channel_to_analyzer(
        self,
        channel: Channel,
        label: str,
        is_used: bool = True
    ) -> 'SettingsBuilder':
        """
        Add a channel to the analyzer.
        
        This is different from add_channel which adds a channel setting interface.
        This method adds a channel to the analyzer's channel list, which affects
        which channels are displayed in the analyzer UI.
        
        Args:
            channel: Channel to add
            label: Label for the channel
            is_used: Whether the channel is used by the analyzer
            
        Returns:
            Self for method chaining
        """
        self._settings.add_channel(channel, label, is_used)
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
    
    Attributes:
        [setting_name]: Access settings directly as attributes
        
    Example:
        ```python
        # Create settings using the builder
        settings = (SettingsBuilder()
                    .add_channel("data", "Data Line")
                    .add_channel("clock", "Clock Line")
                    .build())
                    
        # Access settings by name using attributes
        data_channel = settings.data.channel
        
        # Or using get_setting method
        clock_channel = settings.get_setting("clock").channel
        ```
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
    
    def get_settings_names(self) -> List[str]:
        """
        Get a list of all setting names.
        
        Returns:
            List of setting names
        """
        return list(self._interface_map.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert all settings to a dictionary.
        
        Returns:
            Dictionary mapping setting names to their dictionary representations
        """
        result = {}
        for name, interface in self._interface_map.items():
            result[name] = interface.to_dict()
        return result


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
    
    This function creates a pre-configured set of settings for analyzing UART/serial signals.
    
    Returns:
        Preconfigured settings for UART analysis
        
    Example:
        ```python
        # Create UART settings
        settings = create_uart_settings()
        
        # Configure channels
        settings.tx.channel = Channel(0, 0)
        settings.rx.channel = Channel(0, 1)
        
        # Set baud rate
        settings.baud_rate.value = 115200
        ```
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
    
    This function creates a pre-configured set of settings for analyzing SPI signals.
    
    Returns:
        Preconfigured settings for SPI analysis
        
    Example:
        ```python
        # Create SPI settings
        settings = create_spi_settings()
        
        # Configure channels
        settings.mosi.channel = Channel(0, 0)
        settings.miso.channel = Channel(0, 1)
        settings.clock.channel = Channel(0, 2)
        settings.enable.channel = Channel(0, 3)
        
        # Set to Mode 0
        settings.clock_polarity.value = 0
        settings.clock_phase.value = 0
        ```
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
    
    This function creates a pre-configured set of settings for analyzing I2C signals.
    
    Returns:
        Preconfigured settings for I2C analysis
        
    Example:
        ```python
        # Create I2C settings
        settings = create_i2c_settings()
        
        # Configure channels
        settings.sda.channel = Channel(0, 0)
        settings.scl.channel = Channel(0, 1)
        
        # Use 10-bit addressing
        settings.address_bits.value = 10
        ```
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


def create_1wire_settings() -> BuilderSettings:
    """
    Create settings for a 1-Wire analyzer.
    
    This function creates a pre-configured set of settings for analyzing 1-Wire signals.
    
    Returns:
        Preconfigured settings for 1-Wire analysis
        
    Example:
        ```python
        # Create 1-Wire settings
        settings = create_1wire_settings()
        
        # Configure data channel
        settings.data.channel = Channel(0, 0)
        ```
    """
    builder = SettingsBuilder()
    
    # Add 1-Wire settings
    builder.add_channel("data", "Data", "1-Wire data line")
    
    # Add speed setting
    builder.add_number_list("speed", "Speed", "1-Wire speed mode",
                           [(0, "Standard (15.4 kbps)", "Standard speed"),
                            (1, "Overdrive (125 kbps)", "Overdrive speed")])
    
    # Add display options
    builder.add_bool("show_romid", "Show ROM ID", "Display full ROM ID for each device",
                    checkbox_text="Show ROM ID")
    
    builder.add_bool("show_crc", "Show CRC", "Display CRC validation results",
                    checkbox_text="Show CRC validation")
    
    # Set defaults
    settings = builder.build()
    settings.speed.value = 0  # Standard speed
    settings.show_romid.value = True
    settings.show_crc.value = True
    
    return settings


def create_can_settings() -> BuilderSettings:
    """
    Create settings for a CAN bus analyzer.
    
    This function creates a pre-configured set of settings for analyzing CAN bus signals.
    
    Returns:
        Preconfigured settings for CAN analysis
        
    Example:
        ```python
        # Create CAN settings
        settings = create_can_settings()
        
        # Configure channels
        settings.can_high.channel = Channel(0, 0)
        settings.can_low.channel = Channel(0, 1)
        
        # Set to high speed CAN
        settings.bitrate.value = 1000000  # 1 Mbps
        ```
    """
    builder = SettingsBuilder()
    
    # Add CAN settings
    builder.add_channel("can_high", "CAN High", "CAN High signal line")
    builder.add_channel("can_low", "CAN Low", "CAN Low signal line", allow_none=True)
    
    # Add bitrate setting
    bitrates = [
        (10000, "10 kbps", "Low speed CAN"),
        (20000, "20 kbps", "Low speed CAN"),
        (50000, "50 kbps", "Low speed CAN"),
        (100000, "100 kbps", "Low speed CAN"),
        (125000, "125 kbps", "Medium speed CAN"),
        (250000, "250 kbps", "Medium speed CAN"),
        (500000, "500 kbps", "High speed CAN"),
        (800000, "800 kbps", "High speed CAN"),
        (1000000, "1 Mbps", "High speed CAN"),
    ]
    builder.add_number_list("bitrate", "Bit Rate", "CAN bus bit rate", bitrates)
    
    # Add sample point setting
    builder.add_integer("sample_point", "Sample Point", 
                       "Position of the sample point as a percentage of the bit time",
                       75, 50, 90)
    
    # Add display options
    builder.add_bool("show_extended", "Show Extended IDs", 
                    "Display extended (29-bit) identifiers",
                    checkbox_text="Show extended IDs")
    
    builder.add_bool("show_error_frames", "Show Error Frames", 
                    "Display error frames in the decoded data",
                    checkbox_text="Show error frames")
    
    # Set defaults
    settings = builder.build()
    settings.bitrate.value = 500000  # 500 kbps
    settings.sample_point.value = 75  # 75%
    settings.show_extended.value = True
    settings.show_error_frames.value = True
    
    return settings


class SettingsValidator:
    """
    Utility class for validating analyzer settings.
    
    This class provides methods for validating settings against various criteria,
    such as required channels, valid values, and dependencies between settings.
    
    Example:
        ```python
        # Create a validator for UART settings
        validator = SettingsValidator(uart_settings)
        
        # Add validation rules
        validator.require_channel("tx")
        validator.require_one_of_channels(["rx", "tx"])
        
        # Validate settings
        is_valid, message = validator.validate()
        if not is_valid:
            print(f"Settings not valid: {message}")
        ```
    """
    
    def __init__(self, settings: AnalyzerSettings):
        """
        Initialize a settings validator.
        
        Args:
            settings: The settings to validate
        """
        self.settings = settings
        self._required_channels: List[str] = []
        self._require_one_of: List[List[str]] = []
        self._custom_validators: List[Callable[[], Tuple[bool, str]]] = []
    
    def require_channel(self, name: str) -> 'SettingsValidator':
        """
        Require a channel to be set.
        
        Args:
            name: Name of the channel setting
            
        Returns:
            Self for method chaining
        """
        self._required_channels.append(name)
        return self
    
    def require_one_of_channels(self, names: List[str]) -> 'SettingsValidator':
        """
        Require at least one of the specified channels to be set.
        
        Args:
            names: List of channel setting names
            
        Returns:
            Self for method chaining
        """
        self._require_one_of.append(names)
        return self
    
    def add_custom_validator(self, validator: Callable[[], Tuple[bool, str]]) -> 'SettingsValidator':
        """
        Add a custom validation function.
        
        Args:
            validator: Function that returns (is_valid, message)
            
        Returns:
            Self for method chaining
        """
        self._custom_validators.append(validator)
        return self
    
    def validate(self) -> Tuple[bool, str]:
        """
        Validate the settings.
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Check required channels
        for name in self._required_channels:
            try:
                setting = self.settings.get_setting(name)
                if isinstance(setting, ChannelSetting):
                    channel = setting.channel
                    if channel.device_index < 0 or channel.channel_index < 0:
                        return False, f"Channel '{name}' is required"
            except (KeyError, AttributeError):
                return False, f"Required channel '{name}' not found"
        
        # Check one-of requirements
        for names in self._require_one_of:
            has_one = False
            for name in names:
                try:
                    setting = self.settings.get_setting(name)
                    if isinstance(setting, ChannelSetting):
                        channel = setting.channel
                        if channel.device_index >= 0 and channel.channel_index >= 0:
                            has_one = True
                            break
                except (KeyError, AttributeError):
                    pass
            
            if not has_one:
                channel_list = ", ".join(names)
                return False, f"At least one of these channels is required: {channel_list}"
        
        # Run custom validators
        for validator in self._custom_validators:
            is_valid, message = validator()
            if not is_valid:
                return False, message
        
        return True, ""