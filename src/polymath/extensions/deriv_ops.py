##########################################################################################
# polymath/extensions/deriv_ops.py: Derivative operations
##########################################################################################

from polymath.qube import Qube

__all__ = ['delete_deriv', 'delete_derivs', 'insert_deriv', 'insert_derivs',
           'rename_deriv', 'unique_deriv_name', 'with_deriv', 'without_deriv',
           'without_derivs', 'wod']


def insert_deriv(self, key, deriv, *, override=True):
    """Insert or replace a derivative in this object.

    To prevent recursion, any internal derivatives of a derivative object are stripped
    away. If the object is read-only, then derivatives will also be converted to
    read-only.

    Derivatives cannot be integers. They are converted to floating-point if necessary.

    You cannot replace the pre-existing value of a derivative in a read-only object
    unless you explicit set override=True. However, inserting a new derivative into a
    read-only object is not prevented.

    Parameters:
        key (str): The name of the derivative. Each derivative also becomes accessible
            as an object attribute with "d_d" in front of the name. For example, the
            time-derivative of this object might be keyed by "t", in which case it can
            also be accessed as attribute "d_dt".
        deriv (Qube): The derivative. Derivatives must have the same leading shape and
            the same numerator as the object; denominator items are used for partial
            derivatives.
        override (bool, optional): True to allow the value of a pre-existing
            derivative to be replaced.

    Returns:
        Qube: This object after the derivative has been inserted.

    Raises:
        TypeError: If the derivative class is invalid or if derivatives are disallowed
            for the object class.
        ValueError: If the shape is invalid, or if the key already exists when
            `override` is False.
    """

    if not self._DERIVS_OK:
        raise TypeError(f'derivatives are disallowed in class {type(self).__name__}')

    # Make sure the derivative is compatible with the object
    if not isinstance(deriv, Qube):
        raise TypeError(f'invalid class for derivative "{key}" in '
                        f'{type(self).__name__} object: {type(deriv).__name__}')

    if self._numer != deriv._numer:
        raise ValueError(f'shape mismatch for numerator of derivative "{key}" in '
                         f'{type(self).__name__} object: '
                         f'{deriv._numer}, {self._numer}')

    if self.readonly and (key in self._derivs) and not override:
        raise ValueError(f'derivative "{key}" cannot be replaced in '
                         f'{type(self).__name__} object; is read-only')

    # Prevent recursion, convert to floating point
    deriv = deriv.wod.as_float()

    # Match readonly of parent if necessary
    if self._readonly and not deriv._readonly:
        deriv = deriv.clone(recursive=False).as_readonly()

    # Save in the derivative dictionary and as an attribute
    if deriv._shape != self._shape:
        deriv = deriv.broadcast_to(self._shape)

    self._derivs[key] = deriv
    setattr(self, 'd_d' + key, deriv)

    self._cache.clear()
    return self


def insert_derivs(self, derivs, *, override=False):
    """Insert or replace the derivatives in this object from a dictionary.

    You cannot replace the pre-existing values of any derivative in a read-only object
    unless you explicit set override=True. However, inserting a new derivative into a
    read-only object is not prevented.

    Parameters:
        derivs (dict): The dictionary of derivatives keyed by their names.
        override (bool, optional): True to allow the value of a pre-existing
            derivative to be replaced.

    Returns:
        Qube: This object after the derivatives has been inserted.

    Raises:
        TypeError: If a derivative class is invalid.
        ValueError: If derivatives are disallowed for the object, if a shape is
            invalid, or if a key already exists when `override` is False.
    """

    # Check every insert before proceeding with any
    if self.readonly and not override:
        for key in derivs:
            if key in self._derivs:
                raise ValueError(f'derivative "{key}" cannot be replaced in '
                                 f'{type(self).__name__} object; object is read-only')

    # Insert derivatives
    for key, deriv in derivs.items():
        self.insert_deriv(key, deriv, override=override)

    return self


def delete_deriv(self, key, *, override=False):
    """Delete a single derivative from this object, given the key.

    Derivatives cannot be deleted from a read-only object without explicitly setting
    override=True.

    Parameters:
        key (str): The key of the derivative to remove. If the key does not exist,
            the object is unchanged.
        override (bool, optional): True to allow the deleting of derivatives from a
            read-only object.

    Raises:
        ValueError: If this object is read-only and `override` is False.
    """

    if not override:
        self.require_writeable()

    if key in self._derivs:
        del self._derivs[key]
        del self.__dict__['d_d' + key]

    self._cache.clear()


def delete_derivs(self, *, override=False, preserve=None):
    """Delete all derivatives from this object.

    Derivatives cannot be deleted from a read-only object without explicitly setting
    `override=True`.

    Parameters:
        override (bool, optional): True to allow the deleting of derivatives from a
            read-only object.
        preserve (list, tuple or set, optional): The names of derivatives to retain.
            All others are removed.

    Raises:
        ValueError: If this object is read-only and `override` is False.
    """

    if not override:
        self.require_writeable()

    # If something is being preserved...
    if preserve:

        # Delete derivatives not on the list
        for key in list(self._derivs.keys()):
            if key not in preserve:
                self.delete_deriv(key, override=override)

        return

    # Delete all derivatives
    for key in self._derivs:
        delattr(self, 'd_d' + key)

    self._derivs = {}
    self._cache.clear()


def without_derivs(self, *, preserve=None):
    """A shallow copy of this object without derivatives.

    A read-only object remains read-only, and is cached for later use.

    Parameters:
        preserve (list, tuple, or set, optional): The names of derivatives to retain.
            All others are removed.

    Returns:
        Qube: The copy, with the same subclass as self.
    """

    if not self._derivs:
        return self

    # If something is being preserved...
    if preserve:
        if isinstance(preserve, str):
            preserve = [preserve]

        if not any(p for p in preserve if p in self._derivs):
            return self.wod

        # Create a fast copy with derivatives
        obj = self.clone(recursive=True)

        # Delete derivatives not on the list
        deletions = []
        for key in obj._derivs:
            if key not in preserve:
                deletions.append(key)

        for key in deletions:
            obj.delete_deriv(key, override=True)

        return obj

    # Return a fast copy without derivatives
    return self.wod


@property
def wod(self):
    """A shallow clone without derivatives, cached.

    Read-only objects remain read-only.
    """

    if not self._derivs:
        return self

    if not Qube._DISABLE_CACHE and 'wod' in self._cache:
        return self._cache['wod']

    wod = Qube.__new__(type(self))
    Qube._transfer_attrs(self, wod)

    wod._derivs = {}
    wod._cache = {}
    self._cache['wod'] = wod
    return wod


def without_deriv(self, key):
    """A shallow copy of this object without a particular derivative.

    A read-only object remains read-only.

    Parameters:
        key (str): The key of the derivative to remove.

    Returns:
        Qube: The copy, with the same subclass as self.
    """

    if key not in self._derivs:
        return self

    result = self.clone(recursive=True)
    del result._derivs[key]

    return result


def with_deriv(self, key, value, *, method='insert'):
    """A shallow copy of this object with a derivative inserted or
    added.

    A read-only object remains read-only.

    Parameters:
        key (str): The key of the derivative to insert.
        value (Qube): The value for this derivative.
        method (str): How to insert the derivative, one of these options:`

            * "`insert`": Iinsert the new derivative; raise a ValueError if a
              derivative of the same name already exists.
            * "`replace`":  Replace an existing derivative of the same name.
            * "`add`": Add this derivative to an existing derivative of the same name.

    Returns:
        Qube: The copy, with the same subclass as self.

    Raises:
        ValueError: If `method` is "insert" and a derivative of the given name already
            exists.
    """

    result = self.clone(recursive=True)

    if method not in ('insert', 'replace', 'add'):
        raise ValueError('invalid with_deriv method: ' + repr(method))

    if key in result._derivs:
        if method == 'insert':
            raise ValueError(f'derivative "{key}" already exists in '
                             f'{type(self).__name__} object')
        if method == 'add':
            value = value + result._derivs[key]

    result.insert_deriv(key, value)
    return result


def rename_deriv(self, key, new_key, *, method='insert'):
    """A shallow copy of this object with a derivative renamed.

    A read-only object remains read-only.

    Parameters:
        key (str): The current key of the derivative.
        new_key (str): The new name of the derivative.
        method (str): How to rename the derivative, one of these options:`

            * "`insert`": Iinsert the new derivative; raise a ValueError if a
              derivative of the same name already exists.
            * "`replace`":  Replace an existing derivative of the same name.
            * "`add`": Add this derivative to an existing derivative of the same name.

    Returns:
        Qube: The copy, with the same subclass as self.

    Raises:
        KeyError: If the `key` derivative does not exist.
        ValueError: If `method` is "insert" and a derivative of the given name already
            exists.
    """

    result = self.with_deriv(new_key, self._derivs[key], method=method)
    result = result.without_deriv(key)
    return result


def unique_deriv_name(self, key, *objects):
    """A unique name for a derivative to apply to one or more objects.

    Parameters:
        key (str): The name to use, with a suffix appended if needed.
        *objects (Qube): One or more Qube objects.

    Returns:
        str: The given key, or with a numeric suffix if needed to make it unique.
    """

    # Make a list of all the derivative keys
    all_keys = set(self._derivs.keys())
    for obj in objects:
        if not hasattr(obj, 'derivs'):
            continue
        all_keys |= set(obj._derivs.keys())

    # Return the proposed key if it is unused
    if key not in all_keys:
        return key

    # Otherwise, tack on a number and iterate until the name is unique
    i = 0
    while True:
        unique = key + str(i)
        if unique not in all_keys:
            return unique

        i += 1

##########################################################################################
